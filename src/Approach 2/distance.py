import cv2
import mediapipe as mp
import os
import pygetwindow as gw
from math import sqrt

average_inner_eye_corner_distance = 4
average_nasal_height = 5.0
average_interpupillary_distance = 7

def calculate_focal_length(known_distance, known_width, width_in_frame):
    return (width_in_frame * known_distance) / known_width

def save_focal_length(focal_length, filename='focal_length.txt'):
    with open(filename, 'w') as file:
        file.write(str(focal_length))

def read_focal_length(filename='focal_length.txt'):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return float(file.read())
    return None

def get_screen_resolution():
    screen = gw.getWindowsWithTitle("")[0]
    return screen.width, screen.height

def calculate_distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def detect_and_measure_facial_features(pixel_to_cm_ratio=0.0255):
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye_inner = face_landmarks.landmark[362]
                    right_eye_inner = face_landmarks.landmark[133]
                    left_eye_outer = face_landmarks.landmark[263]
                    right_eye_outer = face_landmarks.landmark[33]
                    bridge_of_nose = face_landmarks.landmark[168]
                    tip_of_nose = face_landmarks.landmark[1]

                    left_eye_inner_px = (int(left_eye_inner.x * width), int(left_eye_inner.y * height))
                    right_eye_inner_px = (int(right_eye_inner.x * width), int(right_eye_inner.y * height))
                    left_eye_outer_px = (int(left_eye_outer.x * width), int(left_eye_outer.y * height))
                    right_eye_outer_px = (int(right_eye_outer.x * width), int(right_eye_outer.y * height))
                    bridge_of_nose_px = (int(bridge_of_nose.x * width), int(bridge_of_nose.y * height))
                    tip_of_nose_px = (int(tip_of_nose.x * width), int(tip_of_nose.y * height))

                    inner_eye_distance_px = calculate_distance(left_eye_inner_px, right_eye_inner_px)
                    nasal_height_px = calculate_distance(bridge_of_nose_px, tip_of_nose_px)
                    interpupillary_distance_px = sqrt(((left_eye_outer_px[0] + left_eye_inner_px[0])/2 - (right_eye_outer_px[0] + right_eye_inner_px[0])/2) ** 2
                                                       + ((left_eye_outer_px[1] + left_eye_inner_px[1])/2 - (right_eye_outer_px[1] + right_eye_inner_px[1])/2) ** 2)

                    inner_eye_distance_cm = inner_eye_distance_px * pixel_to_cm_ratio
                    nasal_height_cm = nasal_height_px * pixel_to_cm_ratio
                    interpupillary_distance_cm = interpupillary_distance_px * pixel_to_cm_ratio

                    #print(f"Inner Eye Corner Distance: {inner_eye_distance_cm:.2f} cm - Average: {average_inner_eye_corner_distance} cm")
                    #print(f"Nasal Height: {nasal_height_cm:.2f} cm - Average: {average_nasal_height} cm")
                    #print(f"Interpupillary Distance: {interpupillary_distance_cm:.2f} cm - Average: {average_interpupillary_distance} cm")

                    cv2.circle(frame, left_eye_inner_px, 3, (0, 255, 0), -1)
                    cv2.circle(frame, right_eye_inner_px, 3, (0, 255, 0), -1)
                    cv2.circle(frame, left_eye_outer_px, 3, (0, 255, 0), -1)
                    cv2.circle(frame, right_eye_outer_px, 3, (0, 255, 0), -1)
                    cv2.circle(frame, bridge_of_nose_px, 3, (0, 255, 0), -1)
                    cv2.circle(frame, tip_of_nose_px, 3, (0, 255, 0), -1)

                    focal_length = read_focal_length()
                    if focal_length is None:
                        known_distance = float(input("Enter the known distance to the reference object (cm): "))
                        focal_length_1 = calculate_focal_length(known_distance, average_inner_eye_corner_distance, inner_eye_distance_cm)
                        focal_length_2 = calculate_focal_length(known_distance, average_nasal_height, nasal_height_cm)
                        focal_length_3 = calculate_focal_length(known_distance, average_interpupillary_distance, interpupillary_distance_cm)
                        focal_length = (focal_length_1+focal_length_2+focal_length_3)/3
                        save_focal_length(focal_length)
                        print(f"Calculated and saved focal length: {focal_length} cms")
                    else:

                        distance_to_user_inner_eye = (average_inner_eye_corner_distance * focal_length) / inner_eye_distance_cm
                        distance_to_user_nasal_height = (average_nasal_height * focal_length) / nasal_height_cm
                        distance_to_user_interpupillary = (average_interpupillary_distance * focal_length) / interpupillary_distance_cm

                        distance = (distance_to_user_inner_eye + distance_to_user_nasal_height + distance_to_user_interpupillary) / 3

                        print(f"Distance to user: {distance:.2f} cm")

            cv2.imshow('Facial Features', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

detect_and_measure_facial_features()