import numpy as np
import cv2

def calculate_gaze_scores(face_2d, last_lx, last_ly, last_rx, last_ry, threshold):
    lx_score, ly_score, rx_score, ry_score = 0, 0, 0, 0
    if (face_2d[243][0] - face_2d[130][0]) != 0:
        lx_score = (face_2d[468][0] - face_2d[130][0]) / (face_2d[243][0] - face_2d[130][0])
        if abs(lx_score - last_lx) < threshold:
            lx_score = (lx_score + last_lx) / 2
        last_lx = lx_score
    if (face_2d[23][1] - face_2d[27][1]) != 0:
        ly_score = (face_2d[468][1] - face_2d[27][1]) / (face_2d[23][1] - face_2d[27][1])
        if abs(ly_score - last_ly) < threshold:
            ly_score = (ly_score + last_ly) / 2
        last_ly = ly_score
    if (face_2d[359][0] - face_2d[463][0]) != 0:
        rx_score = (face_2d[473][0] - face_2d[463][0]) / (face_2d[359][0] - face_2d[463][0])
        if abs(rx_score - last_rx) < threshold:
            rx_score = (rx_score + last_rx) / 2
        last_rx = rx_score
    if (face_2d[253][1] - face_2d[257][1]) != 0:
        ry_score = (face_2d[473][1] - face_2d[257][1]) / (face_2d[253][1] - face_2d[257][1])
        if abs(ry_score - last_ry) < threshold:
            ry_score = (ry_score + last_ry) / 2
        last_ry = ry_score
    return lx_score, ly_score, rx_score, ry_score, last_lx, last_ly, last_rx, last_ry

def calculate_gaze_vectors(l_rvec, r_rvec, lx_score, ly_score, rx_score, ry_score, x_score_multiplier, y_score_multiplier):
    l_gaze_rvec = np.array(l_rvec)
    l_gaze_rvec[2][0] -= (lx_score - 0.5) * x_score_multiplier
    l_gaze_rvec[0][0] += (ly_score - 0.5) * y_score_multiplier
    r_gaze_rvec = np.array(r_rvec)
    r_gaze_rvec[2][0] -= (rx_score - 0.5) * x_score_multiplier
    r_gaze_rvec[0][0] += (ry_score - 0.5) * y_score_multiplier
    l_gaze_rmat, _ = cv2.Rodrigues(l_gaze_rvec)
    r_gaze_rmat, _ = cv2.Rodrigues(r_gaze_rvec)
    return l_gaze_rvec, r_gaze_rvec, l_gaze_rmat, r_gaze_rmat


def project_and_draw_gaze_axes(image, face_2d_head, l_gaze_rvec, r_gaze_rvec, cam_matrix, dist_coeffs, draw_gaze, draw_full_axis):
    axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
    l_corner = face_2d_head[2].astype(np.int32)
    r_corner = face_2d_head[3].astype(np.int32)
    l_gaze_axis, _ = cv2.projectPoints(axis, l_gaze_rvec, np.zeros((3, 1)), cam_matrix, dist_coeffs)
    r_gaze_axis, _ = cv2.projectPoints(axis, r_gaze_rvec, np.zeros((3, 1)), cam_matrix, dist_coeffs)
    if draw_gaze:
        if draw_full_axis:
            cv2.line(image, l_corner, tuple(np.ravel(l_gaze_axis[0]).astype(np.int32)), (1, 0, 0), 3)
            cv2.line(image, l_corner, tuple(np.ravel(l_gaze_axis[1]).astype(np.int32)), (1, 0, 0), 3)
            cv2.line(image, l_corner, tuple(np.ravel(l_gaze_axis[2]).astype(np.int32)), (0, 0, 255), 3)
            cv2.line(image, r_corner, tuple(np.ravel(r_gaze_axis[0]).astype(np.int32)), (1, 0, 0), 3)
            cv2.line(image, r_corner, tuple(np.ravel(r_gaze_axis[1]).astype(np.int32)), (1, 0, 0), 3)
            cv2.line(image, r_corner, tuple(np.ravel(r_gaze_axis[2]).astype(np.int32)), (0, 0, 255), 3)