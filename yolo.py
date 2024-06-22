import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from ultralytics import YOLO


# class EmotionClassifier(nn.Module):
#     def __init__(self):
#         super(EmotionClassifier, self).__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(256 * 2 * 2, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 7),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)

#         return x


# class EyeGazeClassifier(nn.Module):
#     def __init__(self):
#         super(EyeGazeClassifier, self).__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(1, 24, kernel_size=7, padding=1),
#             nn.BatchNorm2d(24),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(24, 24, kernel_size=5, padding=1),
#             nn.BatchNorm2d(24),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(24, 24, kernel_size=3, padding=1),
#             nn.BatchNorm2d(24),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(384, 64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, 4),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)

#         return x


# def predict_emotion(pixels, model, device):
#     emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
#     weights = [0.25, 0.2, 0.3, 0.6, 0.3, 0.6, 0.9]

#     transform = transforms.Compose(
#         [
#             transforms.Resize((48, 48)),
#             transforms.TenCrop(40),
#             transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
#             transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(0.5,), std=(0.5,))(tensor) for tensor in tensors])),
#         ]
#     )

#     pixels = pixels.astype(np.uint8)
#     image = Image.fromarray(pixels)
#     crops = transform(image)

#     model.eval()
#     with torch.no_grad():
#         inputs = crops.to(device)
#         outputs = model(inputs)
#         outputs = outputs.mean(0)
#         probs = torch.softmax(outputs, 0)
#         prob, pred = torch.max(probs, 0)

#     return round(prob.item() * weights[pred], 2)


# def predict_eye_gaze(pixels, model, device):
#     eyes = ["Close", "Forward", "Left", "Right"]

#     transform = transforms.Compose(
#         [
#             transforms.Grayscale(1),
#             transforms.Resize((48, 48)),
#             transforms.TenCrop(40),
#             transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
#             transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(0.5,), std=(0.5,))(tensor) for tensor in tensors])),
#         ]
#     )

#     pixels = pixels.astype(np.uint8)
#     image = Image.fromarray(pixels)
#     crops = transform(image)

#     model.eval()
#     with torch.no_grad():
#         inputs = crops.to(device)
#         outputs = model(inputs)
#         outputs = outputs.mean(0)
#         probs = torch.softmax(outputs, 0)
#         prob, pred = torch.max(probs, 0)

#     if eyes[pred] == "Close":
#         return "Distracted"
#     else:
#         return "Focused"


# device = "cuda" if torch.cuda.is_available() else "cpu"
# emotion_classifier = EmotionClassifier().to(device)
# emotion_classifier.load_state_dict(torch.load("emotion_classifier.pth"))
# eye_gaze_classifier = EyeGazeClassifier().to(device)
# eye_gaze_classifier.load_state_dict(torch.load("eye_gaze_classifier.pth"))
# face_detector = YOLO("face_detector.pt")
# eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     # Capture frame-by-frame
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Perform face detection using YOLO
#     results = face_detector(frame, conf=0.4)
#     boxes = results[0].boxes

#     # Draw bounding box around face
#     if len(boxes.xywh) == 1 and int(boxes.cls.item()) == 2:
#         x, y, w, h = boxes.xywh[0]
#         x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)
#         roi_gray = gray[y : y + h, x : x + w]
#         roi_color = frame[y : y + h, x : x + w]

#         # Predict emotion
#         score = predict_emotion(roi_gray, emotion_classifier, device)
#         if score < 0.15:
#             color = (0, 0, 255)
#             concentration = "Distracted"
#         else:
#             color = (0, 255, 0)
#             concentration = "Focused"
#         cv2.putText(frame, concentration, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#         cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=3)

#         # Detect the eyes
#         eyes = eye_detector.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
#         for ex, ey, ew, eh in eyes:
#             roi_gray_eye = roi_gray[ey : ey + eh, ex : ex + ew]
#             roi_color_eye = roi_color[ey : ey + eh, ex : ex + ew]

#             # # Predict eye gaze
#             # gaze = predict_eye_gaze(roi_gray_eye, eye_gaze_classifier, device)
#             # cv2.putText(roi_color, gaze, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#             # cv2.rectangle(roi_color, pt1=(ex, ey), pt2=(ex + ew, ey + eh), color=(255, 0, 0), thickness=3)

#     elif len(boxes.xywh) == 1 and int(boxes.cls.item()) == 1:
#         x, y, w, h = boxes.xywh[0]
#         x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)
#         cv2.putText(frame, "Face covered", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#         cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 255), thickness=3)
#     elif len(boxes.xywh) == 1 and int(boxes.cls.item()) == 0:
#         x, y, w, h = boxes.xywh[0]
#         x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)
#         cv2.putText(frame, "Distracted", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=3)

#     # Display the resulting frame
#     cv2.imshow("frame", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # When everything is done, release the capture
# cap.release()
# cv2.destroyAllWindows()


face_detector = YOLO("face_detector.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()

    results = face_detector(frame, conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()