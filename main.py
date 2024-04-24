import cv2
import numpy as np

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=3)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        eyes = eye_detector.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, pt1=(ex, ey), pt2=(ex + ew, ey + eh), color=(0, 255, 0), thickness=3)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()