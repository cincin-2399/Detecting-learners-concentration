import cv2
import numpy as np
from mediapipe_utils import initialize_mediapipe, detect_face, detect_face_mesh, draw_face_box
from model_utils import load_xgboost_model, predict_focus
from parameters import initialize_parameters
from gaze_utils import calculate_gaze_scores, calculate_gaze_vectors, project_and_draw_gaze_axes
from head_pose_utils import calculate_head_pose, project_and_draw_head_pose_axes

def main():
    face_detection, face_mesh, mp_drawing, drawing_spec = initialize_mediapipe()
    model = load_xgboost_model('C:/Learning/Materials/Kyhainamba/Computer Vision/Project/src/trained_model_weights/xgb_model.pkl')
    draw_gaze, draw_full_axis, draw_headpose, showing_value, threshold, x_score_multiplier, y_score_multiplier, face_3d, leye_3d, reye_3d = initialize_parameters()

    cap = cv2.VideoCapture(0)
    last_lx, last_rx, last_ly, last_ry = 0, 0, 0, 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        face_detection_results = detect_face(face_detection, image)
        face_mesh_results = detect_face_mesh(face_mesh, image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if face_detection_results.detections:
            for detection in face_detection_results.detections:
                draw_face_box(image, detection)
                
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    face_2d = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in face_landmarks.landmark]
                    face_2d_head = np.array([face_2d[1], face_2d[199], face_2d[33], face_2d[263], face_2d[61], face_2d[291]], dtype=np.float64)
                    
                    lx_score, ly_score, rx_score, ry_score, last_lx, last_ly, last_rx, last_ry = calculate_gaze_scores(face_2d, last_lx, last_ly, last_rx, last_ry, threshold)
                    l_rvec, l_tvec, r_rvec, r_tvec, l_rmat, r_rmat = calculate_head_pose(face_2d_head, image.shape[1], image.shape[0], leye_3d, reye_3d)
                    l_gaze_rvec, r_gaze_rvec, l_gaze_rmat, r_gaze_rmat = calculate_gaze_vectors(l_rvec, r_rvec, lx_score, ly_score, rx_score, ry_score, x_score_multiplier, y_score_multiplier)
                    
                    project_and_draw_head_pose_axes(image, face_2d_head, l_rvec, l_tvec, r_rvec, r_tvec, l_gaze_rvec, r_gaze_rvec, image.shape[1], image.shape[0], draw_headpose, draw_gaze, draw_full_axis)
                    
                    l_head_yaw, l_head_pitch, l_head_roll = cv2.RQDecomp3x3(l_rmat)[0]
                    l_gaze_yaw, l_gaze_pitch, l_gaze_roll = cv2.RQDecomp3x3(l_gaze_rmat)[0]
                    
                    prediction = predict_focus(model, l_head_yaw, l_head_pitch, l_head_roll, l_gaze_yaw, l_gaze_pitch, l_gaze_roll)
                    cv2.putText(image, f"{prediction}", (165, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Can't detect face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Detect learner's concentration", image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()