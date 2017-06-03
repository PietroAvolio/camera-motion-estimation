import motion_estimation
import numpy as np
import cv2
import sys
import time
import operator

features_detection_engine = None

# intrinsix parameter of the camera
camera_matrix = np.asmatrix([[float(1.18848290e+03), 0.0, float(6.42833462e+02)],
				 [0.0, float(1.18459614e+03), float(3.86675542e+02)],
				 [0.0, 0.0, 1.0]])

dist_coeff = np.array([[float(0.23320571), float(-0.11904307), float(0.00389023), float(0.00985417), float(0.55733118)]])
# extrinsic parameters of the camera: rotation and translation matrix

# ported from modules/calib3d/src/five-point.cpp line 373
def computeError(p1, p2, essential_mat):
	# x1 = [p1.x, p1.y, 1.0]
	# x2 = [p2.x, p2.y, 1.0]
	# the point is in 2D, so we assume the z parameter as 1
	# the assumption may lea to wrong projections but since we project both points with the essential matrix
	# we don't account for any projection error (the x, y and z will have the same bias in both points)
	
	# Numpy implementation of transposition and matrix operations is very slow for our computation.
	# I noticed a performance increases from 0.4 fps to 2.4 fps -> 6x

	# E * x1
	Ex1 = [
		essential_mat.item(0)*p1[0] + essential_mat.item(1)*p1[1] + essential_mat.item(2)* 1.0, 
		essential_mat.item(3)*p1[0] + essential_mat.item(4)*p1[1] + essential_mat.item(5)* 1.0, 
		essential_mat.item(6)*p1[0] + essential_mat.item(7)*p1[1] + essential_mat.item(8)* 1.0
	]
	
	# E.transpose() * x1
	Etx2 = [
		essential_mat.item(0)*p2[0] + essential_mat.item(3)*p2[1] + essential_mat.item(6)* 1.0, 
		essential_mat.item(1)*p2[0] + essential_mat.item(4)*p2[1] + essential_mat.item(7)* 1.0, 
		essential_mat.item(2)*p2[0] + essential_mat.item(5)*p2[1] + essential_mat.item(8)* 1.0
	]
	
	# x2.transpose().dot(Ex1)
	x2tEx1 = Ex1[0]*p2[0] + Ex1[1]*p2[1] + Ex1[2]*1.0
	return float(x2tEx1 * x2tEx1 / (Ex1[0] * Ex1[0] + Ex1[1] * Ex1[1] + Etx2[0] * Etx2[0] + Etx2[1] * Etx2[1]))


# i = iteration number; M = number of matched features, B = block size 
def preemptionFunction(i, M, B = 100):
	return int(np.floor(M * np.power(2, np.floor(-i/B))))

def features_detection(image, fps):
	key_points, key_points_desc = features_detection_engine.detectAndCompute(image, None)

	observations = motion_estimation.onNewFeaturesDiscovered(image, key_points, key_points_desc, fps)
	hypotheses = []
	if observations is not None:

		# hypotheses generation
		
		while len(hypotheses) < len(observations):
			# get 6 points (5 points to get the essential matrices + one for validation, so to have a sigle 3x3 matrix instead of n 3x3 matrices)
			rand = np.random.choice(len(observations), 6, replace=False)

			five_features = [observations[x] for x in rand]
			
			f1_points = np.array([x[0].pt for x in five_features])
			f2_points = np.array([x[1].pt for x in five_features])
			
			# get the essential matrix/matrices with the 5 point method
			essential_mat = cv2.findEssentialMat(f1_points[0:5], f2_points[0:5], camera_matrix)
			
			if essential_mat[0] is None:
				continue

			mat = []
			for j in range(0, essential_mat[0].shape[0], 3):
				tmp_mat = np.asmatrix([essential_mat[0][j], essential_mat[0][j+1], essential_mat[0][j+2]])
				mat.append(tmp_mat)

			best = computeError(f1_points[5], f2_points[5], mat[0])
			bestEMat = mat[0]
			
			for j in range(1, len(mat)):
				if computeError(f1_points[5], f2_points[5], mat[j]) < best:
					bestEMat = mat[j]
			
			hypotheses.append(bestEMat)

		print("Generated "+str(len(hypotheses))+" hypotheses")

		# len of hypothesis set
		M = len(hypotheses)
		f = preemptionFunction(1, M)
		scores = []

		# first step of preemptive ransac
		for h in range(0, len(hypotheses)):
			score = computeError(observations[0][0].pt, observations[0][1].pt, hypotheses[h])
			scores.append((h, score))

		lenHypotheses = len(hypotheses)
		
		# scan all observations
		for o in range(1, len(observations)):
			
			# incremental update of the scores
			for h in range(0, lenHypotheses):
				score = computeError(observations[o][0].pt, observations[o][1].pt, hypotheses[h])
				scores[h] = (scores[h][0], scores[h][1]+score)

			# recalculate the preemption function
			f = preemptionFunction(o+1, M)

			# if only one hypothesis exit from the loop
			if f == 1:
				break

			# need to reorder and remove hypotheses with bad score
			if lenHypotheses > f:
				scores = sorted(scores, key=lambda x: x[1], reverse=False)[0:f]
				lenHypotheses = f

		# the correct hypothesis
		hypothesis = hypotheses[scores[0][0]]
		print(hypothesis)
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