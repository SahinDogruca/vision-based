#!/usr/bin/env python3
"""
evaluate.py — LARD V1 Test Set Üzerinde Tam Evaluation

Belirli bir yöntem (method) için tüm LARD_test_synth datasetindeki
görüntüler üzerinde pose estimation çalıştırır ve ortalama (mean),
standart sapma (std), medyan, min, max istatistiklerini hesaplar.

Desteklenen yöntemler (methods):
  gt          — Ground truth köşe noktalarından PnP (ML model gerektirmez)
  yolo        — YOLO keypoint detection + PnP (YOLO model gerekir)
  full        — LoRAT tracking + YOLO + PnP (LoRAT + YOLO model gerekir)

Çıktılar:
  - Konsola tablo formatında genel ve havalimanı bazında istatistikler
  - CSV: evaluation_results.csv (tüm frame sonuçları)
  - CSV: evaluation_summary.csv (istatistik özeti)
  - PNG: hata dağılım grafikleri

Kullanım:
  # GT keypoints ile evaluation (model gerektirmez, sadece CSV + runway_data)
  python evaluate.py \\
      --method gt \\
      --csv_path LARD_test_synth/LARD_test_synth.csv \\
      --runway_data pose_estimation/runway_data.csv \\
      --output_dir evaluation_results

  # YOLO keypoints ile evaluation
  python evaluate.py \\
      --method yolo \\
      --csv_path LARD_test_synth/LARD_test_synth.csv \\
      --lard_images_dir LARD_test_synth/ \\
      --yolo_model models/keypoints/model.pt \\
      --output_dir evaluation_results

  # Sadece belirli havalimanları
  python evaluate.py \\
      --method gt \\
      --csv_path LARD_test_synth/LARD_test_synth.csv \\
      --airports CYYZ CYUL LICJ

  # Belirli senaryolar
  python evaluate.py \\
      --method gt \\
      --csv_path LARD_test_synth/LARD_test_synth.csv \\
      --scenarios CYYZ_05_35 CYUL_06L_35
"""

import os
import sys
import csv
import math
import time
import argparse
import pickle
import warnings

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_WIDTH = 2448
IMG_HEIGHT = 2648

CAMERA_MATRIX = np.array([
    [3407.28, 0.0,     1291.13],
    [0.0,     3506.60, 1321.82],
    [0.0,     0.0,     1.0    ]
], dtype=np.float64)

DIST_COEFFS = np.array([-0.4479, 2.4688, 0.0177, 0.0118, -1.1850], dtype=np.float64)

METERS_PER_NM = 1852


# ---------------------------------------------------------------------------
# Runway data lookup (robust: handles zero-padded vs stripped IDs)
# ---------------------------------------------------------------------------
def find_runway_params(runway_data_path, airport, runway):
    runway_str = str(runway)
    candidates = [runway_str]
    if runway_str[:1].isdigit():
        digits, suffix = '', ''
        for ch in runway_str:
            if ch.isdigit() and not suffix:
                digits += ch
            else:
                suffix += ch
        padded = digits.zfill(2) + suffix
        if padded not in candidates:
            candidates.append(padded)
        stripped = digits.lstrip('0') + suffix
        if stripped and stripped not in candidates:
            candidates.append(stripped)
    with open(runway_data_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Airport'] == airport and row['Runway'] in candidates:
                return row
    return None


# ---------------------------------------------------------------------------
# 3D object points
# ---------------------------------------------------------------------------
def build_3d_object_points(aspect_ratio):
    RUNWAY = (3, 2)
    objp = np.zeros((1, RUNWAY[0] * RUNWAY[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[
        0:RUNWAY[0] / 2:0.5,
        0:RUNWAY[1] * aspect_ratio:aspect_ratio
    ].T.reshape(-1, 2)
    return objp


# ---------------------------------------------------------------------------
# Angle utilities
# ---------------------------------------------------------------------------
def rotation_vector_to_euler_angles(rotation_vector):
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    r = R.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('xyz', degrees=False)
    order = np.array([1, 0, 2])
    return euler_angles[order]


def normalize_angle(angle):
    normalized = angle % (2 * math.pi)
    if normalized > math.pi:
        normalized -= 2 * math.pi
    elif normalized < -math.pi:
        normalized += 2 * math.pi
    return normalized


def interp(x1, x2):
    return (x1 + x2) / 2


# ---------------------------------------------------------------------------
# Camera params loading
# ---------------------------------------------------------------------------
def load_camera_params(repo_root):
    cam_path = os.path.join(repo_root, 'pose_estimation', 'camera_calibration', 'camera_matrix.pkl')
    dist_path = os.path.join(repo_root, 'pose_estimation', 'camera_calibration', 'distortion.pkl')
    if os.path.exists(cam_path) and os.path.exists(dist_path):
        with open(cam_path, 'rb') as f:
            mtx = pickle.load(f)
        with open(dist_path, 'rb') as f:
            dist = pickle.load(f)
        return mtx, dist
    return CAMERA_MATRIX.copy(), DIST_COEFFS.copy()


# ---------------------------------------------------------------------------
# LARD corners → 6 keypoint conversion (repo convention)
# ---------------------------------------------------------------------------
def lard_corners_to_6kp(x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D):
    x_1, y_1 = float(x_C), float(y_C)
    x_2, y_2 = float(x_D), float(y_D)
    if x_1 > x_2:
        x_1, x_2 = x_2, x_1
        y_1, y_2 = y_2, y_1

    x_3, y_3 = float(x_A), float(y_A)
    x_4, y_4 = float(x_B), float(y_B)
    if x_3 > x_4:
        x_3, x_4 = x_4, x_3
        y_3, y_4 = y_4, y_3

    bbox_coord = np.zeros((1, 6, 1, 2), dtype=np.float32)
    bbox_coord[0, 2, 0, 0] = x_1;  bbox_coord[0, 2, 0, 1] = y_1
    bbox_coord[0, 1, 0, 0] = interp(x_1, x_2);  bbox_coord[0, 1, 0, 1] = interp(y_1, y_2)
    bbox_coord[0, 0, 0, 0] = x_2;  bbox_coord[0, 0, 0, 1] = y_2

    bbox_coord[0, 5, 0, 0] = x_3;  bbox_coord[0, 5, 0, 1] = y_3
    bbox_coord[0, 4, 0, 0] = interp(x_3, x_4);  bbox_coord[0, 4, 0, 1] = interp(y_3, y_4)
    bbox_coord[0, 3, 0, 0] = x_4;  bbox_coord[0, 3, 0, 1] = y_4

    return bbox_coord


# ---------------------------------------------------------------------------
# PnP pose estimation
# ---------------------------------------------------------------------------
def estimate_pose_single(keypoints_6, airport, runway, runway_data_path, mtx, dist):
    params = find_runway_params(runway_data_path, airport, runway)
    if params is None:
        return None, None, None, None, False

    runway_width = float(params['Width'])
    aspect_ratio = float(params['Aspect Ratio'])
    yaw_offset = math.radians(float(params['Yaw Offset']))

    objp = build_3d_object_points(aspect_ratio)
    imgp = keypoints_6[0].astype(np.float32)

    try:
        success, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)
    except cv2.error:
        return None, None, None, None, False

    if not success:
        return None, None, None, None, False

    ypr = rotation_vector_to_euler_angles(rvecs)
    ypr[0] += yaw_offset
    ypr[2] += np.pi
    ypr[2] = -normalize_angle(ypr[2])

    yaw_deg = np.degrees(ypr[0])
    pitch_deg = np.degrees(ypr[1])
    roll_deg = np.degrees(ypr[2])

    slant_nm = np.sqrt(
        tvecs[0, 0]**2 + tvecs[1, 0]**2 + tvecs[2, 0]**2
    ) * runway_width / METERS_PER_NM

    return yaw_deg, pitch_deg, roll_deg, slant_nm, True


# ---------------------------------------------------------------------------
# YOLO-based evaluation helpers
# ---------------------------------------------------------------------------
def crop_image_for_yolo(image, x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D, padding_pct=0.2):
    xs = [float(x_A), float(x_B), float(x_C), float(x_D)]
    ys = [float(y_A), float(y_B), float(y_C), float(y_D)]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    half_pad = padding_pct / 2
    h_img, w_img = image.shape[:2]
    left = max(int(x_min - half_pad * bbox_w), 0)
    right = min(int(x_max + half_pad * bbox_w), w_img)
    upper = max(int(y_min - half_pad * bbox_h), 0)
    lower = min(int(y_max + half_pad * bbox_h), h_img)
    crop_w = abs(right - left)
    crop_h = abs(lower - upper)
    cropped = image[upper:lower, left:right]
    cropped_resized = cv2.resize(cropped, (w_img, h_img))
    return cropped_resized, [left, upper, crop_w, crop_h]


def yolo_keypoints_to_6kp(keypoints_xy, crop_data,
                           original_img_width=IMG_WIDTH,
                           original_img_height=IMG_HEIGHT):
    left, upper, width, height = crop_data
    x_1 = int(keypoints_xy[0, 0] / original_img_width * width + left)
    x_2 = int(keypoints_xy[1, 0] / original_img_width * width + left)
    x_3 = int(keypoints_xy[2, 0] / original_img_width * width + left)
    x_4 = int(keypoints_xy[3, 0] / original_img_width * width + left)
    y_1 = int(keypoints_xy[0, 1] / original_img_height * height + upper)
    y_2 = int(keypoints_xy[1, 1] / original_img_height * height + upper)
    y_3 = int(keypoints_xy[2, 1] / original_img_height * height + upper)
    y_4 = int(keypoints_xy[3, 1] / original_img_height * height + upper)

    x_list = np.array([x_1, x_2, x_3, x_4])
    y_list = np.array([y_1, y_2, y_3, y_4])
    sorted_idx = np.argsort(y_list)

    if x_list[sorted_idx[-1]] < x_list[sorted_idx[-2]]:
        bx_1, by_1 = x_list[sorted_idx[-1]], y_list[sorted_idx[-1]]
        bx_2, by_2 = x_list[sorted_idx[-2]], y_list[sorted_idx[-2]]
    else:
        bx_1, by_1 = x_list[sorted_idx[-2]], y_list[sorted_idx[-2]]
        bx_2, by_2 = x_list[sorted_idx[-1]], y_list[sorted_idx[-1]]

    if x_list[sorted_idx[1]] < x_list[sorted_idx[0]]:
        tx_3, ty_3 = x_list[sorted_idx[1]], y_list[sorted_idx[1]]
        tx_4, ty_4 = x_list[sorted_idx[0]], y_list[sorted_idx[0]]
    else:
        tx_3, ty_3 = x_list[sorted_idx[0]], y_list[sorted_idx[0]]
        tx_4, ty_4 = x_list[sorted_idx[1]], y_list[sorted_idx[1]]

    bbox_coord = np.zeros((1, 6, 1, 2), dtype=np.float32)
    bbox_coord[0, 2, 0, 0] = bx_1;  bbox_coord[0, 2, 0, 1] = by_1
    bbox_coord[0, 1, 0, 0] = int(interp(bx_1, bx_2))
    bbox_coord[0, 1, 0, 1] = int(interp(by_1, by_2))
    bbox_coord[0, 0, 0, 0] = bx_2;  bbox_coord[0, 0, 0, 1] = by_2

    bbox_coord[0, 5, 0, 0] = tx_3;  bbox_coord[0, 5, 0, 1] = ty_3
    bbox_coord[0, 4, 0, 0] = int(interp(tx_3, tx_4))
    bbox_coord[0, 4, 0, 1] = int(interp(ty_3, ty_4))
    bbox_coord[0, 3, 0, 0] = tx_4;  bbox_coord[0, 3, 0, 1] = ty_4

    return bbox_coord


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------
def compute_stats(series):
    """Bir pd.Series için mean, std, median, min, max hesapla."""
    return {
        'mean': series.mean(),
        'std': series.std(),
        'median': series.median(),
        'min': series.min(),
        'max': series.max(),
    }


def print_summary_table(results_df, title=""):
    """Tablo formatında ortalama ± std ve diğer istatistikleri yazdır."""
    if title:
        print(f"\n  {'='*74}")
        print(f"  {title}")
        print(f"  {'='*74}")

    metrics = ['err_yaw', 'err_pitch', 'err_roll', 'err_slant']
    labels = ['Yaw Error (°)', 'Pitch Error (°)', 'Roll Error (°)', 'Slant Dist Error']

    print(f"\n  {'Metrik':<20s}  {'Mean ± Std':>18s}  {'Median':>10s}  {'Min':>10s}  {'Max':>10s}  {'N':>6s}")
    print(f"  {'─'*74}")

    for metric, label in zip(metrics, labels):
        if metric not in results_df.columns:
            continue
        vals = results_df[metric].dropna()
        if len(vals) == 0:
            continue
        s = compute_stats(vals.abs())
        print(f"  {label:<20s}  {s['mean']:8.4f} ± {s['std']:<7.4f}  {s['median']:10.4f}  {s['min']:10.4f}  {s['max']:10.4f}  {len(vals):6d}")

    # Success rates
    print(f"\n  Başarı Oranları (tüm açı hataları eşik altında):")
    for thr in [0.1, 0.5, 1.0, 2.0, 5.0]:
        angle_cols = [c for c in ['err_yaw', 'err_pitch', 'err_roll'] if c in results_df.columns]
        if angle_cols:
            mask = True
            for c in angle_cols:
                mask = mask & (results_df[c].abs() < thr)
            rate = mask.sum() / len(results_df) * 100
            print(f"    < {thr:5.1f}°  →  {rate:6.2f}% ({mask.sum()}/{len(results_df)})")


def print_per_airport_table(results_df):
    """Havalimanı bazında istatistik tablosu."""
    print(f"\n  {'='*90}")
    print(f"  Havalimanı/Pist Bazında Sonuçlar")
    print(f"  {'='*90}")

    print(f"\n  {'Airport/Rwy':<14s}  {'N':>4s}  {'Yaw Mean±Std':>16s}  {'Pitch Mean±Std':>16s}  {'Roll Mean±Std':>16s}  {'Slant MAE':>10s}")
    print(f"  {'─'*86}")

    grouped = results_df.groupby(['airport', 'runway'])
    for (ap, rw), group in sorted(grouped):
        n = len(group)
        yaw = group['err_yaw'].abs()
        pitch = group['err_pitch'].abs()
        roll = group['err_roll'].abs()
        slant = group['err_slant'].abs()
        print(f"  {ap+'_'+str(rw):<14s}  {n:4d}  "
              f"{yaw.mean():7.3f}±{yaw.std():6.3f}  "
              f"{pitch.mean():7.3f}±{pitch.std():6.3f}  "
              f"{roll.mean():7.3f}±{roll.std():6.3f}  "
              f"{slant.mean():10.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_evaluation_results(results_df, output_dir, method_name):
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Evaluation Results — Method: {method_name.upper()}  (N={len(results_df)})',
                 fontsize=16, fontweight='bold')

    for ax, col, label, color in [
        (axes[0, 0], 'err_yaw', 'Yaw Error (°)', '#3498db'),
        (axes[0, 1], 'err_pitch', 'Pitch Error (°)', '#2ecc71'),
        (axes[1, 0], 'err_roll', 'Roll Error (°)', '#e74c3c'),
        (axes[1, 1], 'err_slant', 'Slant Distance Error', '#9b59b6'),
    ]:
        if col not in results_df.columns:
            ax.set_visible(False)
            continue
        vals = results_df[col].dropna().abs()
        ax.hist(vals, bins=50, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(vals.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {vals.mean():.4f}')
        ax.axvline(vals.median(), color='orange', linestyle=':', linewidth=2,
                   label=f'Median: {vals.median():.4f}')
        ax.set_title(label, fontsize=13)
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'evaluation_{method_name}_histograms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [INFO] Histogramlar kaydedildi: {path}")

    # Per-airport box plot
    if results_df['airport'].nunique() > 1:
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        fig.suptitle(f'Havalimanı Bazında Hata Dağılımları — {method_name.upper()}',
                     fontsize=14, fontweight='bold')

        for ax, col, label in [
            (axes[0], 'err_yaw', 'Yaw Error (°)'),
            (axes[1], 'err_pitch', 'Pitch Error (°)'),
            (axes[2], 'err_roll', 'Roll Error (°)'),
        ]:
            airports = sorted(results_df['airport'].unique())
            data = [results_df[results_df['airport'] == ap][col].abs().dropna().values
                    for ap in airports]
            ax.boxplot(data, tick_labels=airports, vert=True)
            ax.set_title(label)
            ax.set_ylabel('Absolute Error (°)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f'evaluation_{method_name}_boxplots.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [INFO] Box plot kaydedildi: {path}")


# ===================================================================
# METHOD: GT — Ground truth köşe noktalarından PnP
# ===================================================================
def evaluate_gt(df, runway_data_path, mtx, dist):
    results = []
    total = len(df)
    failed = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        airport = str(row['airport'])
        runway = str(row['runway'])

        kp6 = lard_corners_to_6kp(
            row['x_A'], row['y_A'], row['x_B'], row['y_B'],
            row['x_C'], row['y_C'], row['x_D'], row['y_D']
        )

        est_yaw, est_pitch, est_roll, est_slant, success = estimate_pose_single(
            kp6, airport, runway, runway_data_path, mtx, dist
        )

        if not success:
            failed += 1
            continue

        gt_yaw = float(row['yaw'])
        gt_pitch = float(row['pitch'])
        gt_roll = float(row['roll'])
        gt_slant = float(row['slant_distance'])

        results.append({
            'image': row['image'],
            'airport': airport, 'runway': runway,
            'scenario': str(row.get('scenario', '')),
            'slant_distance': gt_slant,
            'gt_yaw': gt_yaw, 'gt_pitch': gt_pitch, 'gt_roll': gt_roll, 'gt_slant': gt_slant,
            'est_yaw': est_yaw, 'est_pitch': est_pitch, 'est_roll': est_roll, 'est_slant': est_slant,
            'err_yaw': est_yaw - gt_yaw,
            'err_pitch': est_pitch - gt_pitch,
            'err_roll': est_roll - gt_roll,
            'err_slant': est_slant - gt_slant,
        })

        if (idx + 1) % 200 == 0:
            print(f"    İşlenen: {idx + 1}/{total}  (başarısız: {failed})")

    print(f"    Tamamlandı: {len(results)}/{total}  (başarısız: {failed})")
    return pd.DataFrame(results)


# ===================================================================
# METHOD: YOLO — YOLO keypoint detection + PnP
# ===================================================================
def evaluate_yolo(df, runway_data_path, mtx, dist, lard_images_dir, yolo_model_path):
    from ultralytics import YOLO

    model = YOLO(yolo_model_path)
    print(f"    YOLO model yüklendi: {yolo_model_path}")

    results = []
    total = len(df)
    failed = 0
    no_image = 0

    # Temp dir for cropped images
    crop_dir = os.path.join(os.path.dirname(yolo_model_path), '_eval_crops')
    os.makedirs(crop_dir, exist_ok=True)

    for idx, (_, row) in enumerate(df.iterrows()):
        airport = str(row['airport'])
        runway = str(row['runway'])

        # Load image
        image_path = os.path.join(lard_images_dir, str(row['image']))
        if not os.path.exists(image_path):
            no_image += 1
            continue

        image = cv2.imread(image_path)
        if image is None:
            no_image += 1
            continue

        # Crop around GT bbox
        cropped, crop_data = crop_image_for_yolo(
            image, row['x_A'], row['y_A'], row['x_B'], row['y_B'],
            row['x_C'], row['y_C'], row['x_D'], row['y_D']
        )

        crop_path = os.path.join(crop_dir, f'crop_{idx}.jpg')
        cv2.imwrite(crop_path, cropped)

        # Run YOLO
        try:
            yolo_results = model.predict(source=crop_path, verbose=False)
            if (len(yolo_results) == 0 or yolo_results[0].keypoints is None or
                    yolo_results[0].keypoints.xy is None or len(yolo_results[0].keypoints.xy) == 0):
                failed += 1
                continue

            kp_xy = yolo_results[0].keypoints.xy.detach().cpu().numpy()[0]
            if kp_xy.shape[0] < 4:
                failed += 1
                continue

            kp6 = yolo_keypoints_to_6kp(
                kp_xy[:4], crop_data,
                original_img_width=image.shape[1],
                original_img_height=image.shape[0]
            )
        except Exception as e:
            failed += 1
            continue

        est_yaw, est_pitch, est_roll, est_slant, success = estimate_pose_single(
            kp6, airport, runway, runway_data_path, mtx, dist
        )

        if not success:
            failed += 1
            continue

        gt_yaw = float(row['yaw'])
        gt_pitch = float(row['pitch'])
        gt_roll = float(row['roll'])
        gt_slant = float(row['slant_distance'])

        results.append({
            'image': row['image'],
            'airport': airport, 'runway': runway,
            'scenario': str(row.get('scenario', '')),
            'slant_distance': gt_slant,
            'gt_yaw': gt_yaw, 'gt_pitch': gt_pitch, 'gt_roll': gt_roll, 'gt_slant': gt_slant,
            'est_yaw': est_yaw, 'est_pitch': est_pitch, 'est_roll': est_roll, 'est_slant': est_slant,
            'err_yaw': est_yaw - gt_yaw,
            'err_pitch': est_pitch - gt_pitch,
            'err_roll': est_roll - gt_roll,
            'err_slant': est_slant - gt_slant,
        })

        if (idx + 1) % 50 == 0:
            print(f"    İşlenen: {idx + 1}/{total}  (başarısız: {failed}, görüntü yok: {no_image})")

    # Cleanup
    import shutil
    shutil.rmtree(crop_dir, ignore_errors=True)

    print(f"    Tamamlandı: {len(results)}/{total}  (başarısız: {failed}, görüntü yok: {no_image})")
    return pd.DataFrame(results)


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description='LARD V1 Test Set Üzerinde Tam Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--method', type=str, required=True,
                        choices=['gt', 'yolo'],
                        help='Evaluation yöntemi: gt veya yolo')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='LARD CSV dosya yolu (semicolon-separated)')
    parser.add_argument('--runway_data', type=str,
                        default='pose_estimation/runway_data.csv',
                        help='runway_data.csv yolu')
    parser.add_argument('--lard_images_dir', type=str, default='',
                        help='LARD görüntülerinin base dizini (yolo modu için)')
    parser.add_argument('--yolo_model', type=str, default='models/keypoints/model.pt',
                        help='YOLO model yolu (yolo modu için)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Çıktı dizini')
    parser.add_argument('--airports', type=str, nargs='+', default=None,
                        help='Sadece bu havalimanlarını değerlendir')
    parser.add_argument('--scenarios', type=str, nargs='+', default=None,
                        help='Sadece bu senaryoları değerlendir')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maksimum görüntü sayısı (hızlı test için)')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve paths
    runway_data_path = args.runway_data
    if not os.path.isabs(runway_data_path):
        runway_data_path = os.path.join(repo_root, runway_data_path)

    print(f"\n{'='*74}")
    print(f"  LARD V1 Evaluation — Yöntem: {args.method.upper()}")
    print(f"{'='*74}\n")

    # Load CSV
    df = pd.read_csv(args.csv_path, sep=';')
    print(f"  Yüklenen: {len(df)} satır — {args.csv_path}")

    # Filter
    if args.airports:
        df = df[df['airport'].isin(args.airports)]
        print(f"  Havalimanı filtresi: {args.airports} → {len(df)} satır")

    if args.scenarios:
        if 'scenario' in df.columns:
            df = df[df['scenario'].isin(args.scenarios)]
        print(f"  Senaryo filtresi: {args.scenarios} → {len(df)} satır")

    if args.max_images:
        df = df.head(args.max_images)
        print(f"  Maks. görüntü limiti: {args.max_images} → {len(df)} satır")

    if len(df) == 0:
        print("  [HATA] Filtreleme sonrası veri kalmadı!")
        sys.exit(1)

    print(f"\n  Havalimanları: {df['airport'].nunique()}")
    print(f"  Senaryolar: {df['scenario'].nunique() if 'scenario' in df.columns else 'N/A'}")
    print(f"  Toplam görüntü: {len(df)}")

    # Load camera params
    mtx, dist = load_camera_params(repo_root)

    # Run evaluation
    print(f"\n  Evaluation başlıyor...\n")
    t0 = time.time()

    if args.method == 'gt':
        results_df = evaluate_gt(df, runway_data_path, mtx, dist)
    elif args.method == 'yolo':
        yolo_model_path = args.yolo_model
        if not os.path.isabs(yolo_model_path):
            yolo_model_path = os.path.join(repo_root, yolo_model_path)
        if not os.path.exists(yolo_model_path):
            print(f"  [HATA] YOLO model bulunamadı: {yolo_model_path}")
            sys.exit(1)
        results_df = evaluate_yolo(df, runway_data_path, mtx, dist,
                                    args.lard_images_dir, yolo_model_path)

    elapsed = time.time() - t0
    print(f"\n  Süre: {elapsed:.1f} saniye ({elapsed/60:.1f} dakika)")

    if len(results_df) == 0:
        print("  [HATA] Hiçbir sonuç üretilemedi!")
        sys.exit(1)

    # ── Print results ──
    print_summary_table(results_df, f"GENEL SONUÇLAR — {args.method.upper()} (N={len(results_df)})")
    print_per_airport_table(results_df)

    # ── Save CSV ──
    results_csv = os.path.join(args.output_dir, f'evaluation_{args.method}_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\n  [INFO] Tüm sonuçlar: {results_csv}")

    # ── Save summary CSV ──
    summary_rows = []
    for group_name, group_df in [('OVERALL', results_df)] + \
            list(results_df.groupby(['airport', 'runway'])):
        if isinstance(group_name, tuple):
            label = f"{group_name[0]}_{group_name[1]}"
            gdf = group_df
        else:
            label = group_name
            gdf = group_df

        row = {'group': label, 'N': len(gdf)}
        for metric in ['err_yaw', 'err_pitch', 'err_roll', 'err_slant']:
            if metric in gdf.columns:
                vals = gdf[metric].abs()
                row[f'{metric}_mean'] = vals.mean()
                row[f'{metric}_std'] = vals.std()
                row[f'{metric}_median'] = vals.median()
                row[f'{metric}_min'] = vals.min()
                row[f'{metric}_max'] = vals.max()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_dir, f'evaluation_{args.method}_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"  [INFO] Özet istatistikler: {summary_csv}")

    # ── Plots ──
    plot_evaluation_results(results_df, args.output_dir, args.method)

    print(f"\n{'='*74}")
    print(f"  Evaluation tamamlandı.")
    print(f"{'='*74}\n")


if __name__ == '__main__':
    main()
