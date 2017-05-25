import numpy as np
import cv2
import sys

def features_detection(image):
    fast = cv2.FastFeatureDetector_create(60, True)

    kp = fast.detect(image, None)
    cv2.drawKeypoints(image, kp, image, (0, 255, 0))

    cv2.imshow('tracked_features', image)
    return True

def play_video(path):
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()

        if features_detection(frame.copy()):
            cv2.imshow('untracked_features', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

play_video('media/test_1.mp4')