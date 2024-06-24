import mediapipe as mp
import cv2

def initialize_mediapipe():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    return face_detection, face_mesh, mp_drawing, drawing_spec

def detect_face(face_detection, image):
    return face_detection.process(image)

def detect_face_mesh(face_mesh, image):
    return face_mesh.process(image)

def draw_face_box(image, detection):
    img_h, img_w, _ = image.shape
    bounding_box = detection.location_data.relative_bounding_box
    x = int(bounding_box.xmin * img_w)
    y = int(bounding_box.ymin * img_h)
    w = int(bounding_box.width * img_w)
    h = int(bounding_box.height * img_h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)