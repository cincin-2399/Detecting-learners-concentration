import numpy as np

def initialize_parameters():
    draw_gaze = False
    draw_full_axis = False
    draw_headpose = False
    showing_value = False
    threshold = 0.3
    x_score_multiplier = 4
    y_score_multiplier = 4
    face_3d = np.array([
        [0.0, 0.0, 0.0],  # Nose tip
        [0.0, -330.0, -65.0],  # Chin
        [-225.0, 170.0, -135.0],  # Left eye left corner
        [225.0, 170.0, -135.0],  # Right eye right corner
        [-150.0, -150.0, -125.0],  # Left Mouth corner
        [150.0, -150.0, -125.0]  # Right mouth corner
    ], dtype=np.float64)
    leye_3d = face_3d + np.array([225, -175, 135])
    reye_3d = face_3d + np.array([-225, -175, 135])
    return draw_gaze, draw_full_axis, draw_headpose, showing_value, threshold, x_score_multiplier, y_score_multiplier, face_3d, leye_3d, reye_3d