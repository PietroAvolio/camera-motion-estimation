import motion_estimation
import numpy as np
import cv2
import sys
import time

features_detection_engine = None
camera_matrix = np.asmatrix([[float(1.18848290e+03), 0.0, float(6.42833462e+02)],
                 [0.0, float(1.18459614e+03), float(3.86675542e+02)],
                 [0.0, 0.0, 1.0]])

def features_detection(image, fps):
    key_points, key_points_desc = features_detection_engine.detectAndCompute(image, None)

    observations = motion_estimation.onNewFeaturesDiscovered(image, key_points, key_points_desc, fps)
    hypotheses = []
    if observations is not None:
        for i in range(0, len(observations)):
            rand = np.random.choice(len(observations), 5, replace=False)
            five_features = [observations[x] for x in rand]
            f1_points = np.array([x[0].pt for x in five_features])
            f2_points = np.array([x[1].pt for x in five_features])
            essential_mat = cv2.findEssentialMat(f1_points, f2_points, camera_matrix)
            hypotheses.append(essential_mat[0])
        print("Generated "+str(i)+" hypoteses")

    return True


def play_video(path):
    cap = cv2.VideoCapture(path)

    fps_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Reading ", frame_width, "x", frame_height, " @", fps_rate, "fps.")

    framesConsidered = 0
    t1 = time.time()
    fps = 0
    while cap.isOpened():
        ret, frame = cap.read()

        framesConsidered += 1

        elapsed = time.time() - t1
        if elapsed >= 1:
            fps = framesConsidered / elapsed
            t1 = time.time()
            print("FPS: " + str(fps))
            framesConsidered = 0

        if ret:
            if features_detection(frame.copy(), fps):
                cv2.imshow('untracked_features', frame)

        if cv2.waitKey( 1 ) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    features_detection_engine = cv2.ORB_create(nfeatures=650)

    play_video('media/test_1.mp4')