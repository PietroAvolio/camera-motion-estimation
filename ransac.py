import numpy as np
import sys
import cv2
from motion_estimation import scoring_function, camera_matrix, threshold

# we already know the model points, no need to use them as parameter of a function
modelPoints = 5


# it return the overall number of iterations that runsac
# has to perform, accordingly to the accuracy (p) and to
# the percentage of outliers (ep)
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
        return round(num/denom)


# it returns the number of inliers across the whole observation set
# given the hypothesized model.
# ported from modules/calib3d/src/ptsetreg.cpp
# and adapted for our data structure
# findInliers() line 86
def find_inliers(observations, model, threshold):
    #t = threshold * threshold
    inliners = 0
    for observation in observations:
        error = scoring_function(observation[0].pt, observation[1].pt, model)
        if error <= threshold:
            inliners += 1
    return inliners

# RANSAC implementation
# returns the best movement hypothesis (which is an essential matrix)
# ported from modules/calib3d/src/ptsetreg.cpp
# and adapted for our data structure
# run() line 164
def RANSAC_run(observations):
    niters = 1000
    max_good_count = 0

    # default confidence and threshold found on opencv documentation
    confidence = 0.999
    #threshold = 1.0

    # IMPORTANT
    # IMPORTANT
    # The humor of python: 5/6 = 0; 3h of wasted debugging
    # Force this to float, so the division will be a proper float instead of a zero
    # and niters is computed correctly (otherwise ep will be 0)
    count = float(len(observations))
    # IMPORTANT
    # IMPORTANT
    # IMPORTANT

    # number of current RANSAC iteration
    iteration = 0

    while iteration < niters:
        iteration += 1        
        
        # get 5 point at random to generate the essential matrices
        rand = np.random.choice(len(observations), 5, replace=False)
        five_features = [observations[x] for x in rand]
        f1_points = np.array([x[0].pt for x in five_features])
        f2_points = np.array([x[1].pt for x in five_features])
        
        # this calls cv::findEssentialMat() of modules/calib3d/src/five-point.cpp in line 405
        # which calls createLMeDSPointSetRegistrator() of modules/calib3d/src/ptsetreg.cpp in line 
        # which calls LMeDSPointSetRegistrator() in line 273
        # which calls run() in line 277 and on line 305 checks if( count == modelPoints )
        # and count is equal to modelPoints (5), so it calls 
        # runKernel() of modules/calib3d/src/five-point.cpp line 40
        # which performs the needed computation and returns without running either LMeDS nor RANSAC
        essential_mat = cv2.findEssentialMat(f1_points, f2_points, camera_matrix, cv2.RANSAC, 0.999, 1.0)[0]
        
        if essential_mat is None:
            continue

        # scan the n essential matrices
        for i in range(0, essential_mat.shape[0], 3):
            # matrix i
            mat_i = np.asmatrix([essential_mat[i], essential_mat[i + 1], essential_mat[i + 2]])
            good_count = find_inliers(observations, mat_i, threshold)

            # the score is based on the number of inlisers
            # if the score is better, update the status
            if good_count > max(max_good_count, modelPoints-1):
                max_good_count = good_count
                best_mat = mat_i
                # get the new number of overall iterations needed to be run
                niters = ransac_update_num_iters(confidence, (count - good_count)/count, niters)
    
    return best_mat
