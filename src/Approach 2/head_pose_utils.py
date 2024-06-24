import numpy as np
import cv2

def calculate_head_pose(face_2d_head, img_w, img_h, leye_3d, reye_3d):
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    l_rmat, _ = cv2.Rodrigues(l_rvec)
    r_rmat, _ = cv2.Rodrigues(r_rvec)
    return l_rvec, l_tvec, r_rvec, r_tvec, l_rmat, r_rmat

def project_and_draw_head_pose_axes(image, face_2d_head, l_rvec, l_tvec, r_rvec, r_tvec, l_gaze_rvec, r_gaze_rvec, img_w, img_h, draw_headpose, draw_gaze, draw_full_axis):
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
    l_corner = face_2d_head[2].astype(np.int32)
    r_corner = face_2d_head[3].astype(np.int32)
    l_axis, _ = cv2.projectPoints(axis, l_rvec, l_tvec, cam_matrix, dist_coeffs)
    r_axis, _ = cv2.projectPoints(axis, r_rvec, r_tvec, cam_matrix, dist_coeffs)
    if draw_headpose:
        if draw_full_axis:
            cv2.line(image, l_corner, tuple(np.ravel(l_axis[0]).astype(np.int32)), (29, 58, 0), 3)
            cv2.line(image, l_corner, tuple(np.ravel(l_axis[1]).astype(np.int32)), (29, 58, 0), 3)
            cv2.line(image, l_corner, tuple(np.ravel(l_axis[2]).astype(np.int32)), (141, 58, 0), 3)
            cv2.line(image, r_corner, tuple(np.ravel(r_axis[0]).astype(np.int32)), (29, 58, 0), 3)
            cv2.line(image, r_corner, tuple(np.ravel(r_axis[1]).astype(np.int32)), (29, 58, 0), 3)
            cv2.line(image, r_corner, tuple(np.ravel(r_axis[2]).astype(np.int32)), (141, 58, 0), 3)