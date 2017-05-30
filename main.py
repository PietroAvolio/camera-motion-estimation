import motion_estimation
import numpy as np
import cv2
import sys
import time

features_detection_engine = None

def features_detection(image):
    key_points, key_points_desc = features_detection_engine.detectAndCompute(image, None)

    observations = motion_estimation.onNewFeaturesDiscovered(image, key_points, key_points_desc)

    if observations is not None:
    	# matched pair accessible through observations[i] which returns the tuple feature_frame_i, feature_frame_i+1
    	'''
    	for i in ['angle', 'class_id', 'octave', 'pt', 'response', 'size']:
    		print(i, ": ", getattr(observations[0][0], i))
    	
    	for i in ['angle', 'class_id', 'octave', 'pt', 'response', 'size']:
    		print(i, ": ", getattr(observations[0][1], i))
    	sys.exit(0)
    	'''

    cv2.drawKeypoints(image, key_points, image, (0, 255, 0))
    cv2.imshow('tracked_features', image)

    return True

def play_video(path):
    cap = cv2.VideoCapture(path)

    fps_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Reading ", frame_width, "x", frame_height, " @", fps_rate, "fps.")

    framesConsidered = 0
    t1 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if features_detection(frame.copy()):
                cv2.imshow('untracked_features', frame)

        if cv2.waitKey( 1 ) & 0xFF == ord('q'):
            break

        framesConsidered += 1

        elapsed = time.time() - t1
        if elapsed >= 1:
            framesConsidered = framesConsidered/elapsed
            t1 = time.time()
            print("FPS: "+str(framesConsidered))
            framesConsidered = 0


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    features_detection_engine = cv2.ORB_create(nfeatures=650)

    play_video('media/test_1.mp4')