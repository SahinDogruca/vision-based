import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from test_sequence import build_3d_object_points, load_camera_params, rotation_vector_to_euler_angles, normalize_angle

mtx, dist = load_camera_params('.')
objp = build_3d_object_points(56.52)
objp_3d = objp[0]

slant_m = 10000.0
glide_angle_rad = np.radians(3.0)
along_track = slant_m * np.cos(glide_angle_rad)
height = slant_m * np.sin(glide_angle_rad)

yaw_offset_rad = math.radians(42.59)
raw_yaw = math.radians(42.59) - yaw_offset_rad
raw_pitch = np.radians(90.0 - 3.0)
raw_roll = -np.radians(0.0) - np.pi

euler_xyz = np.array([raw_pitch, raw_yaw, raw_roll])
rot = R.from_euler('xyz', euler_xyz)
rot_matrix = rot.as_matrix()
rvec_gt, _ = cv2.Rodrigues(rot_matrix)

runway_center_3d = objp_3d.mean(axis=0)
cam_pos = np.array([
    runway_center_3d[0],
    runway_center_3d[1] - along_track / 59.36,
    -height / 59.36
])
tvec_gt = -rot_matrix @ cam_pos.reshape(3, 1)

imgpoints_proj, _ = cv2.projectPoints(
    objp_3d.reshape(-1, 1, 3).astype(np.float64),
    rvec_gt, tvec_gt, mtx, dist
)
imgpoints_proj = imgpoints_proj.reshape(6, 1, 2).astype(np.float32)

objp_4 = objp[0, [0, 2, 3, 5]]
imgp_4 = imgpoints_proj[[0, 2, 3, 5]]

success, rvecs_list, tvecs_list, _ = cv2.solvePnPGeneric(
    objp_4, imgp_4, mtx, dist, flags=cv2.SOLVEPNP_IPPE
)

for i, (rvec, tvec) in enumerate(zip(rvecs_list, tvecs_list)):
    R_mat, _ = cv2.Rodrigues(rvec)
    ypr = rotation_vector_to_euler_angles(rvec)
    ypr[0] += yaw_offset_rad
    ypr[2] += np.pi
    ypr[2] = -normalize_angle(ypr[2])

    yaw_deg = np.degrees(ypr[0])
    pitch_deg = np.degrees(ypr[1])
    roll_deg = np.degrees(ypr[2])
    print(f"Sol {i}: pitch = {pitch_deg:.2f}, roll = {roll_deg:.2f}, yaw = {yaw_deg:.2f}")
