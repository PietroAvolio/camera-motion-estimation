import motion_estimation
import numpy as np
import cv2
import sys
import time

features_detection_engine = None

# intrinsix parameter of the camera
camera_matrix = np.asmatrix([[float(1.18848290e+03), 0.0, float(6.42833462e+02)],
                 [0.0, float(1.18459614e+03), float(3.86675542e+02)],
                 [0.0, 0.0, 1.0]])

dist_coeff = np.array([[float(0.23320571), float(-0.11904307), float(0.00389023), float(0.00985417), float(0.55733118)]])
# extrinsic parameters of the camera: rotation and translation matrix


# ported from modules/calib3d/src/five-point.cpp line 373
def computeError(p1, p2, essential_mat):
	E = np.asmatrix(essential_mat)

	# the point is in 2D, so we assume the z parameter as 1
	# the assumption may lea to wrong projections but since we project both points with the essential matrix
	# we don't account for any projection error (the x, y and z will have the same bias in both points)
	x1 = np.array([[p1[0]], [p1[1]], [1.0]])
	x2 = np.array([[p2[0]], [p2[1]], [1.0]])
	
	Ex1    = E * x1
	Etx2   = E.transpose() * x2
	x2tEx1 = x2.transpose().dot(Ex1)

	a = Ex1[0] * Ex1[0]
	b = Ex1[1] * Ex1[1]
	c = Etx2[0] * Etx2[0]
	d = Etx2[1] * Etx2[1]

	return float(x2tEx1 * x2tEx1 / (a + b + c + d))

# i = iteration number; M = number of matched features, B = block size 
def preemptionFunction(i, M, B = 100):
	return np.floor(M * np.power(2, np.floor(-i/B)))

def features_detection(image, fps):
    key_points, key_points_desc = features_detection_engine.detectAndCompute(image, None)

    observations = motion_estimation.onNewFeaturesDiscovered(image, key_points, key_points_desc, fps)
    hypotheses = []
    if observations is not None:

    	# hypotheses generation

        for i in range(0, len(observations)+1):
        	# pass 6 points (5 points + one for validation, so to have a 3x3 matrix instead of 3x6 / 3x9)
            rand = np.random.choice(len(observations), 6, replace=False)

            five_features = [observations[x] for x in rand]
            
            f1_points = np.array([x[0].pt for x in five_features])
            f2_points = np.array([x[1].pt for x in five_features])
            
            # get the essential matrix with the 5 point method
            essential_mat = cv2.findEssentialMat(f1_points, f2_points, camera_matrix)
            
            hypotheses.append(essential_mat[0])
            
            #R1, R2, t = cv2.decomposeEssentialMat(essential_mat[0])
            #for j in range(0,5):
            #	print(computeError(f1_points[j], f2_points[j], essential_mat[0]))
            #sys.exit(0)

        print("Generated "+str(i)+" hypoteses")

        # Pre-Emptive RANSAC TODO HERE


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