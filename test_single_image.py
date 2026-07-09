#!/usr/bin/env python3
"""
test_single_image.py — Single Image Pose Estimation on LARD V1

Tests pose estimation on a single LARD image in two modes:
  Mode A (gt):   Uses ground truth keypoints from the LARD CSV
  Mode B (yolo): Uses YOLO keypoint detection within a GT bounding box crop

Usage:
  # Mode A: Ground truth keypoints (no model needed)
  python test_single_image.py --mode gt \
      --csv_path LARD_test_synth/metadata.csv \
      --image_index 0 \
      --runway_data pose_estimation/runway_data.csv

  # Mode B: YOLO prediction
  python test_single_image.py --mode yolo \
      --csv_path LARD_test_synth/metadata.csv \
      --image_index 0 \
      --runway_data pose_estimation/runway_data.csv \
      --yolo_model models/keypoints/model.pt
"""

import os
import sys
import csv
import math
import argparse
import pickle

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_WIDTH = 2448
IMG_HEIGHT = 2648

# Camera intrinsics (matches camera_matrix.pkl for LARD V1 synthetic)
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
    """Look up runway parameters (Width, Aspect Ratio, Yaw Offset) from CSV.

    Tries multiple runway string formats to handle mismatches between
    LARD CSV (integer '5') and runway_data.csv (string '05').
    """
    runway_str = str(runway)
    # Build candidate list: original, zero-padded, stripped
    candidates = [runway_str]
    # Try zero-padded (e.g., '5' → '05')
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

    RUNWAY = (3, 2)
    objp[0,:,:2] = np.mgrid[0:RUNWAY[0]/2:0.5, 0:RUNWAY[1]*AR:AR].T.reshape(-1,2)

    This produces 6 points (x, y, z=0):
      idx 0: (1.0,  0 ) — right end, near edge
      idx 1: (0.5,  0 ) — midpoint, near edge
      idx 2: (0.0,  0 ) — left end, near edge
      idx 3: (1.0, AR ) — right end, far edge
      idx 4: (0.5, AR ) — midpoint, far edge
      idx 5: (0.0, AR ) — left end, far edge
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
    """Convert rotation vector to Euler angles with repo's reordering."""
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    r = R.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('xyz', degrees=False)
    order = np.array([1, 0, 2])
    return euler_angles[order]


def normalize_angle(angle):
    """Normalize angle to [-π, π]."""
    normalized = angle % (2 * math.pi)
    if normalized > math.pi:
        normalized -= 2 * math.pi
    elif normalized < -math.pi:
        normalized += 2 * math.pi
    return normalized


# ---------------------------------------------------------------------------
# Camera matrix loading (try .pkl first, fallback to hardcoded)
# ---------------------------------------------------------------------------
def load_camera_params(repo_root):
    """Load camera matrix and distortion coefficients."""
    cam_path = os.path.join(repo_root, 'pose_estimation', 'camera_calibration', 'camera_matrix.pkl')
    dist_path = os.path.join(repo_root, 'pose_estimation', 'camera_calibration', 'distortion.pkl')

    if os.path.exists(cam_path) and os.path.exists(dist_path):
        with open(cam_path, 'rb') as f:
            mtx = pickle.load(f)
        with open(dist_path, 'rb') as f:
            dist = pickle.load(f)
        print(f"[INFO] Loaded camera params from .pkl files")
        return mtx, dist

    print(f"[INFO] Using hardcoded camera params")
    return CAMERA_MATRIX.copy(), DIST_COEFFS.copy()


# ---------------------------------------------------------------------------
# Keypoint utilities
# ---------------------------------------------------------------------------
def interp(x1, x2):
    return (x1 + x2) / 2


def lard_corners_to_6kp(x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D):
    """
    Convert LARD 4-corner coordinates to the repo's 6-keypoint format.

    Applies the EXACT same sorting/reordering logic from get_ground_truth()
    in pose_estimation_utils.py:
      - C,D = bottom edge (larger y = near camera)
      - A,B = top edge (smaller y = far from camera)
      - Within each edge, enforce left < right (x_1 < x_2)

    Returns: np.array of shape (1, 6, 1, 2) as float32
    """
    # Bottom edge (near): C, D
    x_1, y_1 = float(x_C), float(y_C)
    x_2, y_2 = float(x_D), float(y_D)

    # Ensure x_1 < x_2 for bottom edge
    if x_1 > x_2:
        x_1, x_2 = x_2, x_1
        y_1, y_2 = y_2, y_1

    # Top edge (far): A, B
    x_3, y_3 = float(x_A), float(y_A)
    x_4, y_4 = float(x_B), float(y_B)

    # Ensure x_3 < x_4 for top edge
    if x_3 > x_4:
        x_3, x_4 = x_4, x_3
        y_3, y_4 = y_4, y_3

    # Build the 6-keypoint array matching get_ground_truth() assignment:
    bbox_coord = np.zeros((1, 6, 1, 2), dtype=np.float32)

    # Near edge (bottom, higher y)
    bbox_coord[0, 2, 0, 0] = x_1          # left end
    bbox_coord[0, 2, 0, 1] = y_1
    bbox_coord[0, 1, 0, 0] = interp(x_1, x_2)  # midpoint
    bbox_coord[0, 1, 0, 1] = interp(y_1, y_2)
    bbox_coord[0, 0, 0, 0] = x_2          # right end
    bbox_coord[0, 0, 0, 1] = y_2

    # Far edge (top, lower y)
    bbox_coord[0, 5, 0, 0] = x_3          # left end
    bbox_coord[0, 5, 0, 1] = y_3
    bbox_coord[0, 4, 0, 0] = interp(x_3, x_4)  # midpoint
    bbox_coord[0, 4, 0, 1] = interp(y_3, y_4)
    bbox_coord[0, 3, 0, 0] = x_4          # right end
    bbox_coord[0, 3, 0, 1] = y_4

    return bbox_coord


# ---------------------------------------------------------------------------
# PnP pose estimation (self-contained, mirrors repo logic)
# ---------------------------------------------------------------------------
def estimate_pose_single(keypoints_6, airport, runway, runway_data_path, mtx, dist):
    """
    Estimate pose from 6 keypoints for a single image.

    Args:
        keypoints_6: np.array shape (1, 6, 1, 2) — the 6 keypoints in pixel coords
        airport: ICAO code string
        runway: runway identifier string
        runway_data_path: path to runway_data.csv
        mtx: camera matrix
        dist: distortion coefficients

    Returns:
        yaw_deg, pitch_deg, roll_deg: estimated Euler angles in degrees
        slant_distance_nm: estimated slant distance in nautical miles
        success: bool indicating if solvePnP converged
    """
    params = find_runway_params(runway_data_path, airport, runway)
    if params is None:
        raise ValueError(f"Runway {airport}/{runway} not found in {runway_data_path}")

    runway_width = float(params['Width'])
    aspect_ratio = float(params['Aspect Ratio'])
    yaw_offset = math.radians(float(params['Yaw Offset']))

    # Build 3D template
    objp = build_3d_object_points(aspect_ratio)

    # Image points
    imgp = keypoints_6[0].astype(np.float32)  # shape (6, 1, 2)

    # Solve PnP
    success, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)
    if not success:
        return None, None, None, None, False

    # Convert rotation vector to Euler angles (repo convention)
    ypr = rotation_vector_to_euler_angles(rvecs)

    # Apply corrections (exact copy from repo)
    ypr[0] += yaw_offset           # add runway heading offset to yaw
    ypr[2] += np.pi                # shift roll by π
    ypr[2] = -normalize_angle(ypr[2])  # negate and normalize roll

    # Convert to degrees
    yaw_deg = np.degrees(ypr[0])
    pitch_deg = np.degrees(ypr[1])
    roll_deg = np.degrees(ypr[2])

    # Slant distance in nautical miles
    slant_distance_nm = np.sqrt(
        tvecs[0, 0]**2 + tvecs[1, 0]**2 + tvecs[2, 0]**2
    ) * runway_width / METERS_PER_NM

    return yaw_deg, pitch_deg, roll_deg, slant_distance_nm, True


# ---------------------------------------------------------------------------
# YOLO-based keypoint detection (Mode B)
# ---------------------------------------------------------------------------
def crop_image_from_corners(image, x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D,
                            padding_pct=0.1):
    """
    Crop image around the runway corners with padding.
    Mirrors the repo's cropping approach from get_keypoints.py.

    Returns:
        cropped_resized: PIL Image resized to original dimensions
        crop_data: [image_path, left, upper, crop_width, crop_height]
    """
    from PIL import Image as PILImage

    xs = [float(x_A), float(x_B), float(x_C), float(x_D)]
    ys = [float(y_A), float(y_B), float(y_C), float(y_D)]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    half_pad = padding_pct / 2

    h_img, w_img = image.shape[:2]

    left = int(x_min - half_pad * bbox_w)
    left = max(left, 0)
    right = int(x_max + half_pad * bbox_w)
    right = min(right, w_img)
    upper = int(y_min - half_pad * bbox_h)
    upper = max(upper, 0)
    lower = int(y_max + half_pad * bbox_h)
    lower = min(lower, h_img)

    crop_w = abs(right - left)
    crop_h = abs(lower - upper)

    # Crop and resize to original dimensions (matching repo behavior)
    cropped = image[upper:lower, left:right]
    cropped_resized = cv2.resize(cropped, (w_img, h_img))

    return cropped_resized, [left, upper, crop_w, crop_h]


def yolo_keypoints_to_6kp(keypoints_xy, crop_data,
                           original_img_width=IMG_WIDTH,
                           original_img_height=IMG_HEIGHT):
    """
    Convert YOLO-detected keypoints (4 corners in cropped image coords)
    to the repo's 6-keypoint format in full image coordinates.

    Mirrors format_keypoints() from pose_estimation_utils.py exactly.

    Args:
        keypoints_xy: np.array shape (4, 2) — YOLO detected corners in resized crop coords
        crop_data: [left, upper, crop_width, crop_height]

    Returns:
        bbox_coord: np.array shape (1, 6, 1, 2)
    """
    left, upper, width, height = crop_data

    # Map from resized-crop coordinates back to full image
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

    # Standardize keypoint order (same as format_keypoints)
    sorted_idx = np.argsort(y_list)

    # Bottom two (highest y) = near edge
    if x_list[sorted_idx[-1]] < x_list[sorted_idx[-2]]:
        bx_1 = x_list[sorted_idx[-1]]
        by_1 = y_list[sorted_idx[-1]]
        bx_2 = x_list[sorted_idx[-2]]
        by_2 = y_list[sorted_idx[-2]]
    else:
        bx_1 = x_list[sorted_idx[-2]]
        by_1 = y_list[sorted_idx[-2]]
        bx_2 = x_list[sorted_idx[-1]]
        by_2 = y_list[sorted_idx[-1]]

    # Top two (lowest y) = far edge
    if x_list[sorted_idx[1]] < x_list[sorted_idx[0]]:
        tx_3 = x_list[sorted_idx[1]]
        ty_3 = y_list[sorted_idx[1]]
        tx_4 = x_list[sorted_idx[0]]
        ty_4 = y_list[sorted_idx[0]]
    else:
        tx_3 = x_list[sorted_idx[0]]
        ty_3 = y_list[sorted_idx[0]]
        tx_4 = x_list[sorted_idx[1]]
        ty_4 = y_list[sorted_idx[1]]

    # Build 6-keypoint array
    bbox_coord = np.zeros((1, 6, 1, 2), dtype=np.float32)

    bbox_coord[0, 2, 0, 0] = bx_1
    bbox_coord[0, 2, 0, 1] = by_1
    bbox_coord[0, 1, 0, 0] = int(interp(bx_1, bx_2))
    bbox_coord[0, 1, 0, 1] = int(interp(by_1, by_2))
    bbox_coord[0, 0, 0, 0] = bx_2
    bbox_coord[0, 0, 0, 1] = by_2

    bbox_coord[0, 5, 0, 0] = tx_3
    bbox_coord[0, 5, 0, 1] = ty_3
    bbox_coord[0, 4, 0, 0] = int(interp(tx_3, tx_4))
    bbox_coord[0, 4, 0, 1] = int(interp(ty_3, ty_4))
    bbox_coord[0, 3, 0, 0] = tx_4
    bbox_coord[0, 3, 0, 1] = ty_4

    return bbox_coord


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize_keypoints(image, keypoints_6, gt_corners, save_path):
    """
    Draw detected (green) and GT (red) keypoints on the image and save.

    Args:
        image: BGR image (np.array)
        keypoints_6: shape (1, 6, 1, 2) — estimated keypoints
        gt_corners: dict with x_A, y_A, ..., x_D, y_D
        save_path: output file path
    """
    vis = image.copy()

    # Draw GT corners in red
    gt_pts = [
        (int(gt_corners['x_A']), int(gt_corners['y_A'])),
        (int(gt_corners['x_B']), int(gt_corners['y_B'])),
        (int(gt_corners['x_C']), int(gt_corners['y_C'])),
        (int(gt_corners['x_D']), int(gt_corners['y_D'])),
    ]
    for pt in gt_pts:
        cv2.circle(vis, pt, 12, (0, 0, 255), -1)
        cv2.circle(vis, pt, 14, (255, 255, 255), 2)

    # Draw detected 6 keypoints in green
    labels = ['D(R-near)', 'Mid-near', 'C(L-near)', 'B(R-far)', 'Mid-far', 'A(L-far)']
    colors = [
        (0, 255, 0),    # idx 0
        (0, 200, 200),  # idx 1
        (0, 255, 0),    # idx 2
        (255, 0, 0),    # idx 3
        (200, 200, 0),  # idx 4
        (255, 0, 0),    # idx 5
    ]
    for j in range(6):
        px = int(keypoints_6[0, j, 0, 0])
        py = int(keypoints_6[0, j, 0, 1])
        cv2.circle(vis, (px, py), 8, colors[j], -1)
        cv2.putText(vis, labels[j], (px + 10, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[j], 2)

    # Draw lines connecting near edge and far edge
    near_pts = [
        (int(keypoints_6[0, 2, 0, 0]), int(keypoints_6[0, 2, 0, 1])),
        (int(keypoints_6[0, 1, 0, 0]), int(keypoints_6[0, 1, 0, 1])),
        (int(keypoints_6[0, 0, 0, 0]), int(keypoints_6[0, 0, 0, 1])),
    ]
    far_pts = [
        (int(keypoints_6[0, 5, 0, 0]), int(keypoints_6[0, 5, 0, 1])),
        (int(keypoints_6[0, 4, 0, 0]), int(keypoints_6[0, 4, 0, 1])),
        (int(keypoints_6[0, 3, 0, 0]), int(keypoints_6[0, 3, 0, 1])),
    ]
    for i in range(2):
        cv2.line(vis, near_pts[i], near_pts[i + 1], (0, 255, 128), 2)
        cv2.line(vis, far_pts[i], far_pts[i + 1], (128, 255, 0), 2)
    # Connect sides
    cv2.line(vis, near_pts[0], far_pts[0], (200, 200, 200), 2)
    cv2.line(vis, near_pts[2], far_pts[2], (200, 200, 200), 2)

    cv2.imwrite(save_path, vis)
    print(f"[INFO] Visualization saved to: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Test pose estimation on a single LARD V1 image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--mode', type=str, required=True, choices=['gt', 'yolo'],
                        help='gt = ground truth keypoints, yolo = YOLO detection')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to LARD CSV metadata file (semicolon-separated)')
    parser.add_argument('--image_index', type=int, default=0,
                        help='Row index in the CSV to test (0-based)')
    parser.add_argument('--runway_data', type=str,
                        default='pose_estimation/runway_data.csv',
                        help='Path to runway_data.csv')
    parser.add_argument('--yolo_model', type=str, default='models/keypoints/model.pt',
                        help='Path to fine-tuned YOLOv8n-pose model (mode=yolo)')
    parser.add_argument('--lard_images_dir', type=str, default='',
                        help='Base directory for LARD images (prepended to image column)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory for output files')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Save visualization with keypoints overlaid')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Parse LARD CSV
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  LARD V1 Single Image Pose Estimation — Mode: {args.mode.upper()}")
    print(f"{'='*70}\n")

    df = pd.read_csv(args.csv_path, sep=';')
    if args.image_index >= len(df):
        print(f"[ERROR] image_index {args.image_index} out of range (CSV has {len(df)} rows)")
        sys.exit(1)

    row = df.iloc[args.image_index]

    airport = str(row['airport'])
    runway = str(row['runway']).zfill(2) if isinstance(row['runway'], (int, float)) else str(row['runway'])

    print(f"  Airport:       {airport}")
    print(f"  Runway:        {runway}")
    print(f"  Image:         {row['image']}")
    print(f"  Slant Dist:    {row['slant_distance']}")
    print()

    # Ground truth angles (degrees)
    gt_yaw = float(row['yaw'])
    gt_pitch = float(row['pitch'])
    gt_roll = float(row['roll'])
    gt_slant = float(row['slant_distance'])

    print(f"  GT Yaw:   {gt_yaw:10.4f}°")
    print(f"  GT Pitch: {gt_pitch:10.4f}°")
    print(f"  GT Roll:  {gt_roll:10.4f}°")
    print(f"  GT Slant: {gt_slant:10.4f} (CSV units)")
    print()

    # Corner coordinates
    x_A, y_A = float(row['x_A']), float(row['y_A'])
    x_B, y_B = float(row['x_B']), float(row['y_B'])
    x_C, y_C = float(row['x_C']), float(row['y_C'])
    x_D, y_D = float(row['x_D']), float(row['y_D'])

    print(f"  Corners:  A=({x_A:.0f},{y_A:.0f})  B=({x_B:.0f},{y_B:.0f})  "
          f"C=({x_C:.0f},{y_C:.0f})  D=({x_D:.0f},{y_D:.0f})")

    # ------------------------------------------------------------------
    # 2. Determine repo root for camera params
    # ------------------------------------------------------------------
    repo_root = os.path.dirname(os.path.abspath(__file__))
    mtx, dist = load_camera_params(repo_root)

    # ------------------------------------------------------------------
    # 3. Resolve runway_data path
    # ------------------------------------------------------------------
    runway_data_path = args.runway_data
    if not os.path.isabs(runway_data_path):
        runway_data_path = os.path.join(repo_root, runway_data_path)

    params = find_runway_params(runway_data_path, airport, runway)
    if params is None:
        print(f"\n[ERROR] Runway {airport}/{runway} not found in {runway_data_path}")
        sys.exit(1)

    print(f"\n  Runway Width:   {params['Width']} m")
    print(f"  Aspect Ratio:   {params['Aspect Ratio']}")
    print(f"  Yaw Offset:     {params['Yaw Offset']}°")

    # ==================================================================
    # MODE A: Ground Truth Keypoints
    # ==================================================================
    if args.mode == 'gt':
        print(f"\n{'─'*70}")
        print(f"  MODE A: Ground Truth Keypoints → PnP")
        print(f"{'─'*70}\n")

        # Build 6-keypoint array
        kp6 = lard_corners_to_6kp(x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D)

        print("  6-Keypoint array (idx → x, y):")
        labels = ['D(R-near)', 'Mid-near', 'C(L-near)', 'B(R-far)', 'Mid-far', 'A(L-far)']
        for j in range(6):
            print(f"    [{j}] {labels[j]:12s}  →  ({kp6[0,j,0,0]:.1f}, {kp6[0,j,0,1]:.1f})")

        # Estimate pose
        est_yaw, est_pitch, est_roll, est_slant, success = estimate_pose_single(
            kp6, airport, runway, runway_data_path, mtx, dist
        )

        if not success:
            print("\n  [ERROR] cv2.solvePnP failed to converge!")
            sys.exit(1)

        # Compute errors
        err_yaw = abs(est_yaw - gt_yaw)
        err_pitch = abs(est_pitch - gt_pitch)
        err_roll = abs(est_roll - gt_roll)

        print(f"\n  {'Angle':<10s}  {'GT (°)':>10s}  {'Est (°)':>10s}  {'Error (°)':>10s}  {'Pass':>6s}")
        threshold = 10.0
        print(f"  {'─'*52}")
        print(f"  {'Yaw':<10s}  {gt_yaw:10.4f}  {est_yaw:10.4f}  {err_yaw:10.4f}  {'✓' if err_yaw < threshold else '✗':>6s}")
        print(f"  {'Pitch':<10s}  {gt_pitch:10.4f}  {est_pitch:10.4f}  {err_pitch:10.4f}  {'✓' if err_pitch < threshold else '✗':>6s}")
        print(f"  {'Roll':<10s}  {gt_roll:10.4f}  {est_roll:10.4f}  {err_roll:10.4f}  {'✓' if err_roll < threshold else '✗':>6s}")
        print(f"\n  Slant Distance:  GT={gt_slant:.4f} (CSV units)  Est={est_slant:.4f} NM")

        all_pass = err_yaw < threshold and err_pitch < threshold and err_roll < threshold
        print(f"\n  Overall: {'PASS ✓' if all_pass else 'FAIL ✗'}  (threshold: {threshold}°)")

        # Save CSV
        results_csv = os.path.join(args.output_dir, 'single_image_gt_results.csv')
        pd.DataFrame([{
            'frame': args.image_index,
            'airport': airport,
            'runway': runway,
            'gt_yaw': gt_yaw, 'gt_pitch': gt_pitch, 'gt_roll': gt_roll, 'gt_slant': gt_slant,
            'est_yaw': est_yaw, 'est_pitch': est_pitch, 'est_roll': est_roll, 'est_slant': est_slant,
            'err_yaw': err_yaw, 'err_pitch': err_pitch, 'err_roll': err_roll,
        }]).to_csv(results_csv, index=False)
        print(f"\n  Results saved to: {results_csv}")

        # Visualization
        if args.visualize:
            image_path = row['image']
            if args.lard_images_dir:
                image_path = os.path.join(args.lard_images_dir, image_path)
            elif not os.path.exists(image_path) and args.csv_path:
                image_path = os.path.join(os.path.dirname(args.csv_path), image_path)
            
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                vis_path = os.path.join(args.output_dir, 'single_image_gt_vis.jpg')
                gt_corners = {'x_A': x_A, 'y_A': y_A, 'x_B': x_B, 'y_B': y_B,
                              'x_C': x_C, 'y_C': y_C, 'x_D': x_D, 'y_D': y_D}
                visualize_keypoints(image, kp6, gt_corners, vis_path)
            else:
                print(f"  [WARN] Image not found for visualization: {image_path}")

    # ==================================================================
    # MODE B: YOLO Keypoint Detection
    # ==================================================================
    elif args.mode == 'yolo':
        print(f"\n{'─'*70}")
        print(f"  MODE B: YOLO Keypoint Detection → PnP")
        print(f"{'─'*70}\n")

        # Resolve image path
        image_path = row['image']
        if args.lard_images_dir:
            image_path = os.path.join(args.lard_images_dir, image_path)
        elif not os.path.exists(image_path) and args.csv_path:
            image_path = os.path.join(os.path.dirname(args.csv_path), image_path)

        if not os.path.exists(image_path):
            print(f"  [ERROR] Image not found: {image_path}")
            sys.exit(1)

        image = cv2.imread(image_path)
        if image is None:
            print(f"  [ERROR] Failed to read image: {image_path}")
            sys.exit(1)

        print(f"  Image loaded: {image.shape[1]}x{image.shape[0]}")

        # Crop around GT corners
        cropped, crop_data = crop_image_from_corners(
            image, x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D,
            padding_pct=0.2
        )
        print(f"  Crop: left={crop_data[0]}, upper={crop_data[1]}, "
              f"w={crop_data[2]}, h={crop_data[3]}")

        # Save cropped image for YOLO
        crop_path = os.path.join(args.output_dir, 'cropped_for_yolo.jpg')
        cv2.imwrite(crop_path, cropped)

        # Load YOLO model
        yolo_model_path = args.yolo_model
        if not os.path.isabs(yolo_model_path):
            yolo_model_path = os.path.join(repo_root, yolo_model_path)

        if not os.path.exists(yolo_model_path):
            print(f"  [ERROR] YOLO model not found: {yolo_model_path}")
            sys.exit(1)

        from ultralytics import YOLO
        model = YOLO(yolo_model_path)
        print(f"  YOLO model loaded: {yolo_model_path}")

        # Run inference
        results = model.predict(source=crop_path, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            print("  [ERROR] YOLO detected no keypoints!")
            sys.exit(1)

        kp_data = results[0].keypoints
        if kp_data.xy is None or len(kp_data.xy) == 0:
            print("  [ERROR] YOLO keypoints.xy is empty!")
            sys.exit(1)

        kp_xy = kp_data.xy.detach().cpu().numpy()[0]  # shape (4, 2) or (N, 2)
        print(f"  YOLO detected {kp_xy.shape[0]} keypoints")

        if kp_xy.shape[0] < 4:
            print(f"  [ERROR] Need at least 4 keypoints, got {kp_xy.shape[0]}")
            sys.exit(1)

        # Take first 4 keypoints
        kp_xy_4 = kp_xy[:4]

        # Convert to 6-keypoint format
        kp6 = yolo_keypoints_to_6kp(kp_xy_4, crop_data,
                                     original_img_width=image.shape[1],
                                     original_img_height=image.shape[0])

        print("\n  6-Keypoint array (idx → x, y):")
        labels = ['D(R-near)', 'Mid-near', 'C(L-near)', 'B(R-far)', 'Mid-far', 'A(L-far)']
        for j in range(6):
            print(f"    [{j}] {labels[j]:12s}  →  ({kp6[0,j,0,0]:.1f}, {kp6[0,j,0,1]:.1f})")

        # Also show GT keypoints for comparison
        kp6_gt = lard_corners_to_6kp(x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D)
        print("\n  GT 6-Keypoint array (idx → x, y):")
        for j in range(6):
            print(f"    [{j}] {labels[j]:12s}  →  ({kp6_gt[0,j,0,0]:.1f}, {kp6_gt[0,j,0,1]:.1f})")

        # Estimate pose
        est_yaw, est_pitch, est_roll, est_slant, success = estimate_pose_single(
            kp6, airport, runway, runway_data_path, mtx, dist
        )

        if not success:
            print("\n  [ERROR] cv2.solvePnP failed to converge!")
            sys.exit(1)

        # Compute errors
        err_yaw = abs(est_yaw - gt_yaw)
        err_pitch = abs(est_pitch - gt_pitch)
        err_roll = abs(est_roll - gt_roll)

        print(f"\n  {'Angle':<10s}  {'GT (°)':>10s}  {'Est (°)':>10s}  {'Error (°)':>10s}")
        print(f"  {'─'*46}")
        print(f"  {'Yaw':<10s}  {gt_yaw:10.4f}  {est_yaw:10.4f}  {err_yaw:10.4f}")
        print(f"  {'Pitch':<10s}  {gt_pitch:10.4f}  {est_pitch:10.4f}  {err_pitch:10.4f}")
        print(f"  {'Roll':<10s}  {gt_roll:10.4f}  {est_roll:10.4f}  {err_roll:10.4f}")
        print(f"\n  Slant Distance:  GT={gt_slant:.4f} (CSV units)  Est={est_slant:.4f} NM")

        # Save CSV
        results_csv = os.path.join(args.output_dir, 'single_image_yolo_results.csv')
        pd.DataFrame([{
            'frame': args.image_index,
            'airport': airport,
            'runway': runway,
            'gt_yaw': gt_yaw, 'gt_pitch': gt_pitch, 'gt_roll': gt_roll, 'gt_slant': gt_slant,
            'est_yaw': est_yaw, 'est_pitch': est_pitch, 'est_roll': est_roll, 'est_slant': est_slant,
            'err_yaw': err_yaw, 'err_pitch': err_pitch, 'err_roll': err_roll,
        }]).to_csv(results_csv, index=False)
        print(f"\n  Results saved to: {results_csv}")

        # Visualization
        if args.visualize:
            vis_path = os.path.join(args.output_dir, 'single_image_yolo_vis.jpg')
            gt_corners = {'x_A': x_A, 'y_A': y_A, 'x_B': x_B, 'y_B': y_B,
                          'x_C': x_C, 'y_C': y_C, 'x_D': x_D, 'y_D': y_D}
            visualize_keypoints(image, kp6, gt_corners, vis_path)

    print(f"\n{'='*70}")
    print(f"  Done.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
