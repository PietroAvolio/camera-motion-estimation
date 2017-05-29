import motion_estimation
import cv2

features_detection_engine = None

def features_detection(image):
    key_points, key_points_desc = features_detection_engine.detectAndCompute(image, None)

    motion_estimation.onNewFeaturesDiscovered(image, key_points, key_points_desc)

    #print("Detected ", len(key_points), " features")

    cv2.drawKeypoints(image, key_points, image, (0, 255, 0))
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
    features_detection_engine = cv2.ORB_create(nfeatures=500)

    play_video('media/test_1.mp4')