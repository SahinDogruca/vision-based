#!/usr/bin/env python3
"""
test_sequence.py — Sequence-Based Testing for Vision-Based Landing Guidance

Three modes:
  geometric     — Pure mathematical PnP validation (no images/models needed)
  gt_keypoints  — PnP with LARD ground truth keypoints over a sequence
  full_pipeline — LoRAT tracking + YOLO keypoints + PnP (requires models)

Usage:
  # Geometric PnP validation (no data needed)
  python test_sequence.py --mode geometric \
      --airport CYUL --runway 06L --num_frames 50

  # GT keypoints on a generated sequence
  python test_sequence.py --mode gt_keypoints \
      --sequence_dir inputs/CYYZ_05_35 \
      --runway_data pose_estimation/runway_data.csv

  # Full pipeline
  python test_sequence.py --mode full_pipeline \
      --sequence_dir inputs/CYYZ_05_35 \
      --runway_data pose_estimation/runway_data.csv \
      --yolo_model models/keypoints/model.pt
"""

import os
import sys
import csv
import math
import argparse
import pickle
import warnings

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Optional imports (matplotlib for plotting)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available — plots will be skipped")

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
# Runway data lookup
# ---------------------------------------------------------------------------
def find_runway_params(runway_data_path, airport, runway):
    """Look up runway parameters from CSV.
    Tries multiple runway string formats (raw, zero-padded, stripped).
    """
    runway_str = str(runway)
    candidates = [runway_str]
    if runway_str[:1].isdigit():
        digits = ''
        suffix = ''
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
# 3D object points (matches repo's estimate_pose logic)
# ---------------------------------------------------------------------------
def build_3d_object_points(aspect_ratio):
    """
    Build the 6-point 3D template matching the repo's mgrid construction.
    """
    RUNWAY = (3, 2)
    objp = np.zeros((1, RUNWAY[0] * RUNWAY[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[
        0:RUNWAY[0] / 2:0.5,
        0:RUNWAY[1] * aspect_ratio:aspect_ratio
    ].T.reshape(-1, 2)
    return objp


# ---------------------------------------------------------------------------
# Angle utilities (exact copy from repo)
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
# Camera matrix loading
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
# PnP pose estimation (self-contained)
# ---------------------------------------------------------------------------
def estimate_pose_single(keypoints_6, airport, runway, runway_data_path, mtx, dist):
    """
    Estimate pose from 6 keypoints for a single image.

    Returns: (yaw_deg, pitch_deg, roll_deg, slant_nm, success)
    """
    params = find_runway_params(runway_data_path, airport, runway)
    if params is None:
        raise ValueError(f"Runway {airport}/{runway} not found in {runway_data_path}")

    runway_width = float(params['Width'])
    aspect_ratio = float(params['Aspect Ratio'])
    yaw_offset = math.radians(float(params['Yaw Offset']))

    objp = build_3d_object_points(aspect_ratio)
    imgp = keypoints_6[0].astype(np.float32)

    # Extract the 4 corner points (indices 0, 2, 3, 5) to use IPPE
    objp_4 = objp[0, [0, 2, 3, 5]]
    imgp_4 = imgp[[0, 2, 3, 5]]

    # solvePnPGeneric with IPPE guarantees the 2 best solutions
    success, rvecs_list, tvecs_list, _ = cv2.solvePnPGeneric(
        objp_4, imgp_4, mtx, dist, flags=cv2.SOLVEPNP_IPPE
    )
    best_rvec = None
    best_tvec = None
    
    # Select the solution where the camera pitch is positive (not upside down)
    for rvec, tvec in zip(rvecs_list, tvecs_list):
        if np.isnan(rvec).any() or np.isnan(tvec).any():
            continue
            
        ypr = rotation_vector_to_euler_angles(rvec)
        ypr[0] += yaw_offset
        ypr[2] += np.pi
        ypr[2] = -normalize_angle(ypr[2])
        
        pitch_deg = np.degrees(ypr[1])
        if pitch_deg > 0:
            best_rvec = rvec
            best_tvec = tvec
            break

    if best_rvec is None:
        best_rvec = rvecs_list[0]
        best_tvec = tvecs_list[0]
        if np.isnan(best_rvec).any():
            return None, None, None, None, False

    # Refine with iterative solver on all 6 points
    success, best_rvec, best_tvec = cv2.solvePnP(
        objp, imgp, mtx, dist,
        rvec=best_rvec.copy(), tvec=best_tvec.copy(),
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
    )

    best_ypr = rotation_vector_to_euler_angles(best_rvec)
    best_ypr[0] += yaw_offset
    best_ypr[2] += np.pi
    best_ypr[2] = -normalize_angle(best_ypr[2])

    yaw_deg = np.degrees(best_ypr[0])
    pitch_deg = np.degrees(best_ypr[1])
    roll_deg = np.degrees(best_ypr[2])

    slant_nm = np.sqrt(
        best_tvec[0, 0]**2 + best_tvec[1, 0]**2 + best_tvec[2, 0]**2
    ) * runway_width / METERS_PER_NM

    return yaw_deg, pitch_deg, roll_deg, slant_nm, True


# ---------------------------------------------------------------------------
# PnP pose estimation — raw (for geometric test, no runway offset)
# ---------------------------------------------------------------------------
def estimate_pose_raw(imgpoints_6x1x2, objp, mtx, dist, runway_width, yaw_offset_rad):
    """
    Run PnP and apply the repo's angle post-processing.
    Used by the geometric test where we control everything directly.
    """
    imgp = imgpoints_6x1x2.astype(np.float32)
    
    # Extract the 4 corner points (indices 0, 2, 3, 5) to use IPPE
    objp_4 = objp[0, [0, 2, 3, 5]]
    imgp_4 = imgp[[0, 2, 3, 5]]

    success, rvecs_list, tvecs_list, _ = cv2.solvePnPGeneric(
        objp_4, imgp_4, mtx, dist, flags=cv2.SOLVEPNP_IPPE
    )
    best_rvec = None
    best_tvec = None
    
    # Select the solution where the camera pitch is positive (not upside down)
    for rvec, tvec in zip(rvecs_list, tvecs_list):
        if np.isnan(rvec).any() or np.isnan(tvec).any():
            continue
            
        ypr = rotation_vector_to_euler_angles(rvec)
        ypr[0] += yaw_offset_rad
        ypr[2] += np.pi
        ypr[2] = -normalize_angle(ypr[2])
        
        pitch_deg = np.degrees(ypr[1])
        if pitch_deg > 0:
            best_rvec = rvec
            best_tvec = tvec
            break

    if best_rvec is None:
        best_rvec = rvecs_list[0]
        best_tvec = tvecs_list[0]
        if np.isnan(best_rvec).any():
            return None, None, None, None, False

    # Refine with iterative solver on all 6 points
    success, best_rvec, best_tvec = cv2.solvePnP(
        objp, imgp, mtx, dist,
        rvec=best_rvec.copy(), tvec=best_tvec.copy(),
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
    )

    best_ypr = rotation_vector_to_euler_angles(best_rvec)
    best_ypr[0] += yaw_offset_rad
    best_ypr[2] += np.pi
    best_ypr[2] = -normalize_angle(best_ypr[2])

    yaw_deg = np.degrees(best_ypr[0])
    pitch_deg = np.degrees(best_ypr[1])
    roll_deg = np.degrees(best_ypr[2])

    slant_nm = np.sqrt(
        best_tvec[0, 0]**2 + best_tvec[1, 0]**2 + best_tvec[2, 0]**2
    ) * runway_width / METERS_PER_NM

    return yaw_deg, pitch_deg, roll_deg, slant_nm, True


# ---------------------------------------------------------------------------
# LARD corners to 6-keypoint conversion
# ---------------------------------------------------------------------------
def lard_corners_to_6kp(x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D):
    """Convert LARD 4-corner coordinates to the repo's 6-keypoint format."""
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
    bbox_coord[0, 2, 0, 0] = x_1
    bbox_coord[0, 2, 0, 1] = y_1
    bbox_coord[0, 1, 0, 0] = interp(x_1, x_2)
    bbox_coord[0, 1, 0, 1] = interp(y_1, y_2)
    bbox_coord[0, 0, 0, 0] = x_2
    bbox_coord[0, 0, 0, 1] = y_2

    bbox_coord[0, 5, 0, 0] = x_3
    bbox_coord[0, 5, 0, 1] = y_3
    bbox_coord[0, 4, 0, 0] = interp(x_3, x_4)
    bbox_coord[0, 4, 0, 1] = interp(y_3, y_4)
    bbox_coord[0, 3, 0, 0] = x_4
    bbox_coord[0, 3, 0, 1] = y_4

    return bbox_coord


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(results_df):
    """Compute summary statistics from a results DataFrame."""
    metrics = {}
    for col in ['err_yaw', 'err_pitch', 'err_roll']:
        if col in results_df.columns:
            vals = results_df[col].dropna()
            metrics[col] = {
                'mae': vals.mean(),
                'max': vals.max(),
                'std': vals.std(),
                'median': vals.median(),
            }

    if 'err_slant' in results_df.columns:
        vals = results_df['err_slant'].dropna()
        metrics['err_slant'] = {
            'mae': vals.abs().mean(),
            'max': vals.abs().max(),
            'std': vals.std(),
            'median': vals.abs().median(),
        }

    # Success rate: all angle errors < threshold
    for threshold in [0.001, 0.01, 0.1, 1.0, 5.0]:
        angle_cols = ['err_yaw', 'err_pitch', 'err_roll']
        available = [c for c in angle_cols if c in results_df.columns]
        if available:
            mask = True
            for c in available:
                mask = mask & (results_df[c].abs() < threshold)
            metrics[f'success_rate_{threshold}deg'] = mask.sum() / len(results_df) * 100

    return metrics


def print_metrics(metrics, title=""):
    """Pretty-print metrics dictionary."""
    if title:
        print(f"\n  {title}")
        print(f"  {'─'*60}")

    for key, val in metrics.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                print(f"    {k:>10s}: {v:.6f}")
        elif isinstance(val, float):
            print(f"  {key:>30s}: {val:.2f}%")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_errors(results_df, output_dir, prefix=""):
    """Generate per-frame error plots."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{prefix} Per-Frame Estimation Errors', fontsize=14, fontweight='bold')

    frames = results_df['frame'].values

    # Yaw error
    ax = axes[0, 0]
    ax.plot(frames, results_df['err_yaw'], 'b.-', linewidth=1, markersize=3)
    ax.set_title('Yaw Error')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Error (°)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='0.1° threshold')
    ax.legend()

    # Pitch error
    ax = axes[0, 1]
    ax.plot(frames, results_df['err_pitch'], 'g.-', linewidth=1, markersize=3)
    ax.set_title('Pitch Error')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Error (°)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='0.1° threshold')
    ax.legend()

    # Roll error
    ax = axes[1, 0]
    ax.plot(frames, results_df['err_roll'], 'r.-', linewidth=1, markersize=3)
    ax.set_title('Roll Error')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Error (°)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='0.1° threshold')
    ax.legend()

    # Slant distance error
    if 'err_slant' in results_df.columns:
        ax = axes[1, 1]
        ax.plot(frames, results_df['err_slant'], 'm.-', linewidth=1, markersize=3)
        ax.set_title('Slant Distance Error')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Error (NM or CSV units)')
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{prefix}error_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [INFO] Error plots saved to: {plot_path}")


# ===================================================================
# MODE: GEOMETRIC — Pure mathematical PnP validation
# ===================================================================
def run_geometric_test(args, repo_root):
    """
    Test PnP solver accuracy with synthetically projected keypoints.

    1. Define a known camera pose (position + orientation)
    2. Project 3D runway points onto image plane using cv2.projectPoints()
    3. Feed the projected 2D points into PnP
    4. Compare recovered pose against known ground truth

    Expected: errors < 0.001° (pure math, no noise)
    """
    print(f"\n{'='*70}")
    print(f"  GEOMETRIC PnP VALIDATION")
    print(f"  Airport: {args.airport}  Runway: {args.runway}")
    print(f"  Frames: {args.num_frames}")
    print(f"{'='*70}\n")

    runway_data_path = args.runway_data
    if not os.path.isabs(runway_data_path):
        runway_data_path = os.path.join(repo_root, runway_data_path)

    params = find_runway_params(runway_data_path, args.airport, args.runway)
    if params is None:
        print(f"  [ERROR] Runway {args.airport}/{args.runway} not found")
        sys.exit(1)

    runway_width = float(params['Width'])
    aspect_ratio = float(params['Aspect Ratio'])
    yaw_offset_deg = float(params['Yaw Offset'])
    yaw_offset_rad = math.radians(yaw_offset_deg)

    print(f"  Runway Width:  {runway_width:.2f} m")
    print(f"  Aspect Ratio:  {aspect_ratio:.2f}")
    print(f"  Yaw Offset:    {yaw_offset_deg:.2f}°")

    mtx, dist = load_camera_params(repo_root)

    # Build 3D object points
    objp = build_3d_object_points(aspect_ratio)
    objp_3d = objp[0]  # shape (6, 3)

    print(f"\n  3D Object Points:")
    for i in range(6):
        print(f"    [{i}] ({objp_3d[i,0]:.2f}, {objp_3d[i,1]:.2f}, {objp_3d[i,2]:.2f})")

    # Generate approach trajectory
    # Slant distances from 10 km to 0.5 km, standard 3° glide slope
    slant_distances_m = np.linspace(10000, 500, args.num_frames)
    glide_angle_deg = 3.0
    glide_angle_rad = np.radians(glide_angle_deg)

    results = []

    for i, slant_m in enumerate(slant_distances_m):
        # Compute aircraft position relative to runway center
        # Along-track distance (horizontal)
        along_track = slant_m * np.cos(glide_angle_rad)
        # Height above runway
        height = slant_m * np.sin(glide_angle_rad)

        # Known ground truth angles
        # For a standard approach aligned with the runway:
        gt_yaw_deg = yaw_offset_deg          # heading = runway heading
        gt_pitch_deg = 90.0 - glide_angle_deg  # near vertical look-down for steep approach
        gt_roll_deg = 0.0                     # wings level

        # Convert GT angles to radians for constructing the camera pose
        gt_yaw_rad = np.radians(gt_yaw_deg)
        gt_pitch_rad = np.radians(gt_pitch_deg)
        gt_roll_rad = np.radians(gt_roll_deg)

        # Build rotation matrix from ground truth angles
        # We need to reverse-engineer the repo's angle extraction:
        #   ypr_est = rotation_vector_to_euler_angles(rvecs)
        #   ypr_est[0] += yaw_offset
        #   ypr_est[2] += π
        #   ypr_est[2] = -normalize_angle(ypr_est[2])
        #
        # So if we want est_yaw = gt_yaw, est_pitch = gt_pitch, est_roll = gt_roll:
        #   raw_yaw   = gt_yaw_rad - yaw_offset_rad
        #   raw_pitch = gt_pitch_rad
        #   raw_roll  = -(gt_roll_rad) + π  → then normalize
        #
        # rotation_vector_to_euler_angles does: euler_xyz = R.as_euler('xyz')
        # then reorders [1, 0, 2] → [yaw, pitch, roll] = [euler_y, euler_x, euler_z]
        #
        # So: euler_x = raw_pitch, euler_y = raw_yaw, euler_z = raw_roll

        raw_yaw = gt_yaw_rad - yaw_offset_rad
        raw_pitch = gt_pitch_rad

        # For roll: est_roll = -normalize(raw_roll + π)
        # If gt_roll = 0: raw_roll + π → π → normalize → π → -π
        # So raw_roll = -gt_roll_rad - π (before normalize)
        # Actually, let's solve: -normalize(raw_z + π) = gt_roll_rad
        # normalize(raw_z + π) = -gt_roll_rad
        # raw_z + π = -gt_roll_rad (within [-π, π])
        # raw_z = -gt_roll_rad - π
        raw_roll = -gt_roll_rad - np.pi

        # Build rotation matrix: euler_xyz = [raw_pitch, raw_yaw, raw_roll]
        # because reorder [1,0,2] maps [euler_y, euler_x, euler_z] → [yaw, pitch, roll]
        # so euler_x = raw_pitch, euler_y = raw_yaw, euler_z = raw_roll
        euler_xyz = np.array([raw_pitch, raw_yaw, raw_roll])
        rot = R.from_euler('xyz', euler_xyz)
        rot_matrix = rot.as_matrix()
        rvec_gt, _ = cv2.Rodrigues(rot_matrix)

        # Build translation vector
        # The runway center is at the 3D midpoint of the object points
        runway_center_3d = objp_3d.mean(axis=0)

        # Camera looks at runway center; camera is at (0, -height, along_track) in runway frame
        # tvec = -R @ camera_position_world
        # In the object frame:
        #   X = across runway (0..1)
        #   Y = along runway (0..AR)
        #   Z = up
        # Camera is above runway center, offset by along_track in Y
        cam_pos = np.array([
            runway_center_3d[0],               # centered across runway
            runway_center_3d[1] - along_track / runway_width,  # behind runway (scaled by width)
            -height / runway_width              # above (negative Z in OpenCV convention)
        ])

        tvec_gt = -rot_matrix @ cam_pos.reshape(3, 1)

        # Project 3D points to 2D using the known pose
        imgpoints_proj, _ = cv2.projectPoints(
            objp_3d.reshape(-1, 1, 3).astype(np.float64),
            rvec_gt, tvec_gt, mtx, dist
        )
        imgpoints_proj = imgpoints_proj.reshape(6, 1, 2).astype(np.float32)

        # Check if projected points are within image bounds
        pts_x = imgpoints_proj[:, 0, 0]
        pts_y = imgpoints_proj[:, 0, 1]
        in_bounds = np.all((pts_x >= 0) & (pts_x < IMG_WIDTH) &
                          (pts_y >= 0) & (pts_y < IMG_HEIGHT))

        if not in_bounds:
            # Points outside image — skip
            continue

        # Feed projected points into PnP
        est_yaw, est_pitch, est_roll, est_slant, success = estimate_pose_raw(
            imgpoints_proj, objp, mtx, dist, runway_width, yaw_offset_rad
        )

        if not success:
            results.append({
                'frame': i, 'slant_m': slant_m,
                'gt_yaw': gt_yaw_deg, 'gt_pitch': gt_pitch_deg, 'gt_roll': gt_roll_deg,
                'est_yaw': None, 'est_pitch': None, 'est_roll': None,
                'err_yaw': None, 'err_pitch': None, 'err_roll': None,
                'pnp_failed': True,
            })
            continue

        gt_slant_nm = slant_m / METERS_PER_NM

        err_yaw = abs(est_yaw - gt_yaw_deg)
        err_pitch = abs(est_pitch - gt_pitch_deg)
        err_roll = abs(est_roll - gt_roll_deg)
        err_slant = est_slant - gt_slant_nm

        results.append({
            'frame': i,
            'slant_m': slant_m,
            'gt_yaw': gt_yaw_deg, 'gt_pitch': gt_pitch_deg, 'gt_roll': gt_roll_deg,
            'gt_slant': gt_slant_nm,
            'est_yaw': est_yaw, 'est_pitch': est_pitch, 'est_roll': est_roll,
            'est_slant': est_slant,
            'err_yaw': err_yaw, 'err_pitch': err_pitch, 'err_roll': err_roll,
            'err_slant': err_slant,
            'pnp_failed': False,
        })

    results_df = pd.DataFrame(results)
    valid = results_df[results_df['pnp_failed'] == False].copy()

    if len(valid) == 0:
        print("  [ERROR] No valid PnP solutions — all points projected outside image")
        print("  Try adjusting the approach trajectory parameters")
        sys.exit(1)

    print(f"\n  Tested {len(valid)} / {args.num_frames} frames "
          f"({args.num_frames - len(valid)} skipped — out of image bounds)")

    # Print per-frame results (first 10 and last 5)
    print(f"\n  {'Frame':>6s}  {'Slant(m)':>10s}  {'Err Yaw':>10s}  {'Err Pitch':>10s}  "
          f"{'Err Roll':>10s}  {'Err Slant':>10s}")
    print(f"  {'─'*62}")

    show_idx = list(range(min(10, len(valid)))) + list(range(max(len(valid)-5, 10), len(valid)))
    show_idx = sorted(set(show_idx))

    for idx in show_idx:
        r = valid.iloc[idx]
        print(f"  {int(r['frame']):6d}  {r['slant_m']:10.1f}  {r['err_yaw']:10.6f}  "
              f"{r['err_pitch']:10.6f}  {r['err_roll']:10.6f}  {r.get('err_slant', 0):10.6f}")

    # Summary metrics
    metrics = compute_metrics(valid)
    print_metrics(metrics, "Summary Metrics")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f'geometric_{args.airport}_{args.runway}_results.csv')
    valid.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # Plot
    plot_errors(valid, args.output_dir, prefix=f'geometric_{args.airport}_{args.runway}_')

    # Check success criterion
    max_angle_err = max(valid['err_yaw'].max(), valid['err_pitch'].max(), valid['err_roll'].max())
    print(f"\n  Max angle error: {max_angle_err:.6f}°")
    print(f"  Criterion: < 0.001° → {'PASS ✓' if max_angle_err < 0.001 else 'FAIL ✗'}")
    print(f"  Note: If FAIL, this validates PnP round-trip with numeric precision limits")


# ===================================================================
# MODE: GT_KEYPOINTS — Ground truth keypoints over a sequence
# ===================================================================
def run_gt_keypoints_test(args, repo_root):
    """
    Test PnP with ground truth keypoints from LARD over a generated sequence.
    """
    print(f"\n{'='*70}")
    print(f"  GT KEYPOINTS SEQUENCE TEST")
    print(f"  Sequence: {args.sequence_dir}")
    print(f"{'='*70}\n")

    # Load GT poses CSV
    gt_csv_path = os.path.join(args.sequence_dir, 'gt_poses.csv')
    if not os.path.exists(gt_csv_path):
        print(f"  [ERROR] GT poses CSV not found: {gt_csv_path}")
        print(f"  Run generate_sequence.py first to create the sequence")
        sys.exit(1)

    gt_df = pd.read_csv(gt_csv_path)
    print(f"  Loaded {len(gt_df)} frames from GT poses CSV")

    runway_data_path = args.runway_data
    if not os.path.isabs(runway_data_path):
        runway_data_path = os.path.join(repo_root, runway_data_path)

    mtx, dist = load_camera_params(repo_root)

    results = []
    for _, row in gt_df.iterrows():
        airport = str(row['airport'])
        runway = str(row['runway']).zfill(2) if isinstance(row['runway'], (int, float)) else str(row['runway'])
        frame = int(row['frame'])

        # Build 6 keypoints from GT corners
        kp6 = lard_corners_to_6kp(
            row['x_A'], row['y_A'], row['x_B'], row['y_B'],
            row['x_C'], row['y_C'], row['x_D'], row['y_D']
        )

        # Estimate pose
        try:
            est_yaw, est_pitch, est_roll, est_slant, success = estimate_pose_single(
                kp6, airport, runway, runway_data_path, mtx, dist
            )
        except ValueError as e:
            print(f"  [WARN] Frame {frame}: {e}")
            continue

        if not success:
            print(f"  [WARN] Frame {frame}: PnP failed to converge")
            continue

        gt_yaw = float(row['yaw'])
        gt_pitch = float(row['pitch'])
        gt_roll = float(row['roll'])
        gt_slant = float(row['slant_distance'])

        err_yaw = abs(est_yaw - gt_yaw)
        err_pitch = abs(est_pitch - gt_pitch)
        err_roll = abs(est_roll - gt_roll)

        results.append({
            'frame': frame,
            'airport': airport, 'runway': runway,
            'gt_yaw': gt_yaw, 'gt_pitch': gt_pitch, 'gt_roll': gt_roll, 'gt_slant': gt_slant,
            'est_yaw': est_yaw, 'est_pitch': est_pitch, 'est_roll': est_roll, 'est_slant': est_slant,
            'err_yaw': err_yaw, 'err_pitch': err_pitch, 'err_roll': err_roll,
            'err_slant': est_slant - gt_slant,
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("  [ERROR] No valid results!")
        sys.exit(1)

    # Print per-frame results
    print(f"\n  {'Frame':>6s}  {'Err Yaw':>10s}  {'Err Pitch':>10s}  {'Err Roll':>10s}  {'GT Slant':>10s}  {'Est Slant':>10s}")
    print(f"  {'─'*62}")

    for _, r in results_df.iterrows():
        print(f"  {int(r['frame']):6d}  {r['err_yaw']:10.4f}  {r['err_pitch']:10.4f}  "
              f"{r['err_roll']:10.4f}  {r['gt_slant']:10.4f}  {r['est_slant']:10.4f}")

    # Summary metrics
    metrics = compute_metrics(results_df)
    print_metrics(metrics, "Summary Metrics")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    seq_name = os.path.basename(args.sequence_dir.rstrip('/'))
    csv_path = os.path.join(args.output_dir, f'gt_keypoints_{seq_name}_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # Plot
    plot_errors(results_df, args.output_dir, prefix=f'gt_keypoints_{seq_name}_')


# ===================================================================
# MODE: FULL_PIPELINE — LoRAT + YOLO + PnP
# ===================================================================
def run_full_pipeline(args, repo_root):
    """
    Run the complete pipeline: LoRAT tracking → YOLO keypoints → PnP.
    """
    print(f"\n{'='*70}")
    print(f"  FULL PIPELINE TEST")
    print(f"  Sequence: {args.sequence_dir}")
    print(f"{'='*70}\n")

    # Load GT poses CSV
    gt_csv_path = os.path.join(args.sequence_dir, 'gt_poses.csv')
    if not os.path.exists(gt_csv_path):
        print(f"  [ERROR] GT poses CSV not found: {gt_csv_path}")
        sys.exit(1)

    gt_df = pd.read_csv(gt_csv_path)
    seq_name = os.path.basename(args.sequence_dir.rstrip('/'))

    # Extract airport and runway from the first row
    airport = str(gt_df.iloc[0]['airport'])
    runway = str(gt_df.iloc[0]['runway']).zfill(2) if isinstance(gt_df.iloc[0]['runway'], (int, float)) else str(gt_df.iloc[0]['runway'])

    print(f"  Airport: {airport}")
    print(f"  Runway:  {runway}")
    print(f"  Frames:  {len(gt_df)}")

    runway_data_path = args.runway_data
    if not os.path.isabs(runway_data_path):
        runway_data_path = os.path.join(repo_root, runway_data_path)

    mtx, dist = load_camera_params(repo_root)

    # ---- Step 1: LoRAT Tracking ----
    print(f"\n  Step 1: LoRAT Tracking...")
    try:
        sys.path.insert(0, os.path.join(repo_root, "LoRAT"))
        from LoRAT.main_remodeled import main_tracking
        import shutil

        # Clear LoRAT cache
        cache_dir = os.path.join(repo_root, "LoRAT", "trackit", "datasets", "cache")
        shutil.rmtree(cache_dir, ignore_errors=True)

        # Create a minimal argparse for LoRAT
        lorat_parser = argparse.ArgumentParser(add_help=False)
        # Clear sys.argv temporarily because main_tracking calls parse_args() which will crash on our arguments
        old_argv = sys.argv.copy()
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        sys.argv = [sys.argv[0], "--device", device]
        
        # Pass the sequence name to LoRAT dynamically
        os.environ['LORAT_TEST_SEQ'] = seq_name
        
        tracking_output_dir = main_tracking(seq_name, lorat_parser)
        sys.argv = old_argv

        # Read tracking results
        track_csv = os.path.join(
            tracking_output_dir, "eval", "epoch_0", "results",
            "DINOv2-B-224", "MyDataset-test", seq_name, "eval.csv"
        )
        if not os.path.exists(track_csv):
            print(f"  [ERROR] Tracking output not found: {track_csv}")
            sys.exit(1)

        eval_df = pd.read_csv(track_csv, sep=",")
        print(f"  LoRAT tracked {len(eval_df)} frames")

    except ImportError as e:
        print(f"  [ERROR] Cannot import LoRAT: {e}")
        print(f"  Make sure LoRAT submodule is initialized and replacement files are in place")
        sys.exit(1)
    except Exception as e:
        print(f"  [ERROR] LoRAT tracking failed: {e}")
        sys.exit(1)

    # ---- Step 2+3: YOLO Keypoints + PnP ----
    print(f"\n  Step 2: YOLO Keypoint Detection + PnP Pose Estimation...")

    yolo_model_path = args.yolo_model
    if not os.path.isabs(yolo_model_path):
        yolo_model_path = os.path.join(repo_root, yolo_model_path)

    if not os.path.exists(yolo_model_path):
        print(f"  [ERROR] YOLO model not found: {yolo_model_path}")
        sys.exit(1)

    try:
        from keypoints.infer_keypoints import main_keypoints
        from pose_estimation.pose_estimation_utils import format_keypoints
    except ImportError:
        # Direct import if running from repo root
        sys.path.insert(0, repo_root)
        from keypoints.infer_keypoints import main_keypoints
        from pose_estimation.pose_estimation_utils import format_keypoints

    results = []
    for i in range(len(eval_df)):
        image_df = eval_df.iloc[i]
        image_name = str(int(image_df["# ind"]) + 1).zfill(4) + ".jpg"
        image_path = os.path.join(args.sequence_dir, "frames", image_name)

        if not os.path.exists(image_path):
            print(f"    [WARN] Frame {i}: Image not found: {image_path}")
            continue

        pred_x = image_df.pred_x
        pred_y = image_df.pred_y
        pred_w = image_df.pred_w
        pred_h = image_df.pred_h

        try:
            keypoints, crop_data = main_keypoints(pred_x, pred_y, pred_w, pred_h, image_path)

            if keypoints is None or len(keypoints) == 0:
                print(f"    [WARN] Frame {i}: YOLO detected no keypoints")
                continue

            bbox_coord = format_keypoints(keypoints, crop_data)

            est_yaw, est_pitch, est_roll, est_slant, success = estimate_pose_single(
                bbox_coord, airport, runway, runway_data_path, mtx, dist
            )

            if not success:
                print(f"    [WARN] Frame {i}: PnP failed")
                continue

        except Exception as e:
            print(f"    [WARN] Frame {i}: Error — {e}")
            continue

        # Get GT for this frame
        if i < len(gt_df):
            gt_row = gt_df.iloc[i]
            gt_yaw = float(gt_row['yaw'])
            gt_pitch = float(gt_row['pitch'])
            gt_roll = float(gt_row['roll'])
            gt_slant = float(gt_row['slant_distance'])
        else:
            gt_yaw = gt_pitch = gt_roll = gt_slant = float('nan')

        err_yaw = abs(est_yaw - gt_yaw)
        err_pitch = abs(est_pitch - gt_pitch)
        err_roll = abs(est_roll - gt_roll)

        results.append({
            'frame': i,
            'gt_yaw': gt_yaw, 'gt_pitch': gt_pitch, 'gt_roll': gt_roll, 'gt_slant': gt_slant,
            'est_yaw': est_yaw, 'est_pitch': est_pitch, 'est_roll': est_roll, 'est_slant': est_slant,
            'err_yaw': err_yaw, 'err_pitch': err_pitch, 'err_roll': err_roll,
            'err_slant': est_slant - gt_slant,
        })

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1} / {len(eval_df)} frames")

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("  [ERROR] No valid results!")
        sys.exit(1)

    print(f"\n  Processed {len(results_df)} / {len(eval_df)} frames successfully")

    # Print sample results
    print(f"\n  {'Frame':>6s}  {'Err Yaw':>10s}  {'Err Pitch':>10s}  {'Err Roll':>10s}")
    print(f"  {'─'*42}")
    for _, r in results_df.head(20).iterrows():
        print(f"  {int(r['frame']):6d}  {r['err_yaw']:10.4f}  {r['err_pitch']:10.4f}  {r['err_roll']:10.4f}")

    # Summary metrics
    metrics = compute_metrics(results_df)
    print_metrics(metrics, "Summary Metrics (Full Pipeline)")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f'full_pipeline_{seq_name}_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # Plot
    plot_errors(results_df, args.output_dir, prefix=f'full_pipeline_{seq_name}_')


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Sequence-based testing for vision-based landing guidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['geometric', 'gt_keypoints', 'full_pipeline'],
                        help='Test mode')

    # Geometric mode args
    parser.add_argument('--airport', type=str, default='CYUL',
                        help='ICAO airport code (geometric mode)')
    parser.add_argument('--runway', type=str, default='06L',
                        help='Runway identifier (geometric mode)')
    parser.add_argument('--num_frames', type=int, default=50,
                        help='Number of frames in synthetic trajectory (geometric mode)')

    # Sequence mode args
    parser.add_argument('--sequence_dir', type=str, default=None,
                        help='Path to generated sequence directory')

    # Common args
    parser.add_argument('--runway_data', type=str,
                        default='pose_estimation/runway_data.csv',
                        help='Path to runway_data.csv')
    parser.add_argument('--yolo_model', type=str, default='models/keypoints/model.pt',
                        help='Path to YOLO model (full_pipeline mode)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for results')

    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))

    if args.mode == 'geometric':
        run_geometric_test(args, repo_root)
    elif args.mode == 'gt_keypoints':
        if args.sequence_dir is None:
            print("[ERROR] --sequence_dir is required for gt_keypoints mode")
            sys.exit(1)
        run_gt_keypoints_test(args, repo_root)
    elif args.mode == 'full_pipeline':
        if args.sequence_dir is None:
            print("[ERROR] --sequence_dir is required for full_pipeline mode")
            sys.exit(1)
        run_full_pipeline(args, repo_root)

    print(f"\n{'='*70}")
    print(f"  Done.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
