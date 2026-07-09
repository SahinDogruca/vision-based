#!/usr/bin/env python3
"""
generate_sequence.py — Generate synthetic sequences from LARD V1 images

Reads the LARD CSV, groups images by scenario, sorts by slant_distance
(descending = far to near), and creates sequence directory structures
compatible with the repo's pipeline.

Output structure for each scenario:
  inputs/
    SCENARIO_NAME/
      frames/
        0001.jpg
        0002.jpg
        ...
      groundtruth.txt          (bbox per frame, comma-separated XYXY)
      gt_poses.csv             (ground truth angles + distances)

Usage:
  python generate_sequence.py \
      --csv_path LARD_test_synth/metadata.csv \
      --output_dir inputs/ \
      --lard_images_dir LARD_test_synth/

  # Generate only specific scenarios
  python generate_sequence.py \
      --csv_path LARD_test_synth/metadata.csv \
      --output_dir inputs/ \
      --lard_images_dir LARD_test_synth/ \
      --scenarios CYYZ_05_35 CYUL_06L_35

  # Use symlinks instead of copying
  python generate_sequence.py \
      --csv_path LARD_test_synth/metadata.csv \
      --output_dir inputs/ \
      --lard_images_dir LARD_test_synth/ \
      --symlink
"""

import os
import sys
import shutil
import argparse

import numpy as np
import pandas as pd


def compute_bbox_from_corners(x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D):
    """
    Compute bounding box in XYXY format from 4 corner coordinates.

    Returns: (x_min, y_min, x_max, y_max) as integers
    """
    xs = [float(x_A), float(x_B), float(x_C), float(x_D)]
    ys = [float(y_A), float(y_B), float(y_C), float(y_D)]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def generate_sequence(scenario_df, scenario_name, output_dir, lard_images_dir,
                      use_symlink=False):
    """
    Generate a single sequence directory from a scenario DataFrame.

    Args:
        scenario_df: DataFrame with rows for this scenario, sorted by slant_distance desc
        scenario_name: e.g. 'CYYZ_05_35'
        output_dir: base output directory (e.g. 'inputs/')
        lard_images_dir: base directory where LARD images are stored
        use_symlink: if True, create symlinks instead of copying images
    """
    seq_dir = os.path.join(output_dir, scenario_name)
    frames_dir = os.path.join(seq_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    n_frames = len(scenario_df)

    # Prepare groundtruth.txt lines and GT poses
    gt_lines = []
    gt_poses = []
    frame_count = 0

    for idx, (_, row) in enumerate(scenario_df.iterrows()):
        frame_num = idx + 1
        frame_name = f"{frame_num:04d}.jpg"
        frame_dest = os.path.join(frames_dir, frame_name)

        # Resolve source image path
        image_rel = str(row['image'])
        image_src = os.path.join(lard_images_dir, image_rel)

        if not os.path.exists(image_src):
            print(f"  [WARN] Image not found, skipping: {image_src}")
            continue

        # Copy or symlink the image
        if os.path.exists(frame_dest):
            os.remove(frame_dest)

        if use_symlink:
            os.symlink(os.path.abspath(image_src), frame_dest)
        else:
            shutil.copy2(image_src, frame_dest)

        # Compute bounding box from corners
        x_min, y_min, x_max, y_max = compute_bbox_from_corners(
            row['x_A'], row['y_A'], row['x_B'], row['y_B'],
            row['x_C'], row['y_C'], row['x_D'], row['y_D']
        )
        gt_lines.append(f"{x_min},{y_min},{x_max},{y_max}")

        # Collect GT pose data
        gt_poses.append({
            'frame': frame_num,
            'image': image_rel,
            'airport': str(row['airport']),
            'runway': str(row['runway']),
            'yaw': float(row['yaw']),
            'pitch': float(row['pitch']),
            'roll': float(row['roll']),
            'slant_distance': float(row['slant_distance']),
            'x_A': float(row['x_A']), 'y_A': float(row['y_A']),
            'x_B': float(row['x_B']), 'y_B': float(row['y_B']),
            'x_C': float(row['x_C']), 'y_C': float(row['y_C']),
            'x_D': float(row['x_D']), 'y_D': float(row['y_D']),
            'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max,
        })

        frame_count += 1

    if frame_count == 0:
        print(f"  [WARN] No frames generated for scenario {scenario_name}")
        shutil.rmtree(seq_dir, ignore_errors=True)
        return 0

    # Write groundtruth.txt
    gt_path = os.path.join(seq_dir, 'groundtruth.txt')
    with open(gt_path, 'w') as f:
        f.write('\n'.join(gt_lines) + '\n')

    # Write GT poses CSV
    poses_path = os.path.join(seq_dir, 'gt_poses.csv')
    pd.DataFrame(gt_poses).to_csv(poses_path, index=False)

    print(f"  [OK] {scenario_name}: {frame_count} frames → {seq_dir}")
    return frame_count


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic sequences from LARD V1 images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to LARD CSV metadata file (semicolon-separated)')
    parser.add_argument('--output_dir', type=str, default='inputs',
                        help='Base output directory for sequences')
    parser.add_argument('--lard_images_dir', type=str, default='',
                        help='Base directory for LARD images')
    parser.add_argument('--scenarios', type=str, nargs='+', default=None,
                        help='Generate only these scenarios (space-separated). '
                             'If not specified, generates all scenarios.')
    parser.add_argument('--symlink', action='store_true', default=False,
                        help='Use symlinks instead of copying images')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames per sequence (for quick testing)')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  LARD V1 Sequence Generator")
    print(f"{'='*70}\n")

    # Read LARD CSV
    df = pd.read_csv(args.csv_path, sep=';')
    print(f"  Loaded {len(df)} rows from {args.csv_path}")

    # Group by scenario
    if 'scenario' in df.columns:
        group_col = 'scenario'
    elif 'original_dataset' in df.columns:
        group_col = 'original_dataset'
    else:
        # Fallback: construct scenario from airport + runway
        df['_scenario'] = df['airport'].astype(str) + '_' + df['runway'].astype(str)
        group_col = '_scenario'

    grouped = df.groupby(group_col)
    all_scenarios = sorted(grouped.groups.keys())

    print(f"  Found {len(all_scenarios)} scenarios: {', '.join(all_scenarios[:10])}"
          f"{'...' if len(all_scenarios) > 10 else ''}")

    # Filter scenarios if specified
    if args.scenarios:
        selected = [s for s in args.scenarios if s in all_scenarios]
        missing = [s for s in args.scenarios if s not in all_scenarios]
        if missing:
            print(f"  [WARN] Scenarios not found: {', '.join(missing)}")
        if not selected:
            print(f"  [ERROR] No valid scenarios to process")
            sys.exit(1)
    else:
        selected = all_scenarios

    print(f"  Processing {len(selected)} scenario(s)\n")

    os.makedirs(args.output_dir, exist_ok=True)

    total_frames = 0
    total_sequences = 0

    for scenario_name in selected:
        scenario_df = grouped.get_group(scenario_name).copy()

        # Sort by slant_distance descending (far → near)
        scenario_df = scenario_df.sort_values('slant_distance', ascending=False)

        # Limit frames if requested
        if args.max_frames is not None:
            scenario_df = scenario_df.head(args.max_frames)

        n = generate_sequence(
            scenario_df, str(scenario_name), args.output_dir,
            args.lard_images_dir, use_symlink=args.symlink
        )
        if n > 0:
            total_frames += n
            total_sequences += 1

    print(f"\n{'─'*70}")
    print(f"  Summary: {total_sequences} sequences, {total_frames} total frames")
    print(f"  Output directory: {os.path.abspath(args.output_dir)}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
