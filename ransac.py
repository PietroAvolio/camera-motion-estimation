import numpy as np
import sys
import cv2
from motion_estimation import camera_matrix, scoring_function, check_subset

# we already know the model points, no need to use them as parameter of a function
modelPoints = 5


# ported from modules/calib3d/src/ptsetreg.cpp
# and adapted for our data structure

# RANSACUpdateNumIters() line 53
def ransac_update_num_iters(p, ep, maxIters):
    p = max(p, 0.0)
    p = min(p, 1.0)
    ep = max(ep, 0.0)
    ep = min(ep, 1.0)

    num = max(1.0 - p, sys.float_info.min)
    denom = 1.0 - np.power(1.0 - ep, modelPoints)

    if denom < sys.float_info.min:
        return 0

    num = np.log(num)
    denom = np.log(denom)

    if denom >= 0 or -num >= maxIters*(-denom):
        return maxIters
    else:
        return np.floor(num/denom)


# findInliers() line 86
def find_inliers(observations, model, threshold):
    t = threshold * threshold
    inliners = 0
    for observation in observations:
        error = scoring_function(observation[0].pt, observation[1].pt, model)
        if error <= t:
            inliners += 1
    return inliners

# run() line 164
def RANSAC_run(observations):
    niters = 1000
    max_good_count = 0

    # default confidence and threshold found on opencv documentation
    confidence = 0.999
    threshold = 1.0

    count = len(observations)
    iteration = 0

    while iteration < niters:
        
        while True:
            rand = np.random.choice(len(observations), 5, replace=False)
            five_features = [observations[x] for x in rand]
            f1_points = np.array([x[0].pt for x in five_features])
            f2_points = np.array([x[1].pt for x in five_features])
            if check_subset(f1_points, f2_points):
                break

        # this calls cv::findEssentialMat() of modules/calib3d/src/five-point.cpp in line 405
        # which calls createLMeDSPointSetRegistrator() of modules/calib3d/src/ptsetreg.cpp in line 
        # which calls LMeDSPointSetRegistrator() in line 273
        # which calls run() in line 277 and on line 305 checks if( count == modelPoints )
        # and count is equal to modelPoints (5), so it calls 
        # runKernel() of modules/calib3d/src/five-point.cpp line 40
        # which performs the needed computation and returns without running either LMeDS nor RANSAC
        essential_mat = cv2.findEssentialMat(f1_points, f2_points, camera_matrix)[0]

        if essential_mat is None:
            iteration += 1
            continue

        for i in range(0, essential_mat.shape[0], 3):
            mat_i = np.asmatrix([essential_mat[i], essential_mat[i + 1], essential_mat[i + 2]])
            good_count = find_inliers(observations, mat_i, threshold)

            if good_count > max(max_good_count, modelPoints-1):
                max_good_count = good_count
                best_mat = mat_i
                niters = ransac_update_num_iters(confidence, (count - good_count)/count, niters)
        iteration += 1
    return best_mat
