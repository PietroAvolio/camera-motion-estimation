import numpy as np
import cv2

def features_detection(image):
    fd_engine = cv2.FastFeatureDetector_create(87, True)

    kp = fd_engine.detect(image, None)
    cv2.drawKeypoints(image, kp, image, (0, 255, 0))

    print("Detected " , len(kp), " features /w ", fd_engine.getThreshold())

    cv2.imshow('tracked_features', image)
    return True

def play_video(path):
    cap = cv2.VideoCapture(path)

    fps_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Reading ", frame_width, "x", frame_height, " @", fps_rate, "fps.")

    while cap.isOpened():
        ret, frame = cap.read()

        if features_detection(frame.copy()):
            cv2.imshow('untracked_features', frame)

        if cv2.waitKey( 1 ) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_video('media/test_1.mp4')