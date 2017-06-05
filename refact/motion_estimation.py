import cv2
import numpy as np

# intrinsix parameter of the camera
camera_matrix = np.asmatrix([[float(1.18848290e+03), 0.0, float(6.42833462e+02)],
                             [0.0, float(1.18459614e+03), float(3.86675542e+02)],
                             [0.0, 0.0, 1.0]])

dist_coeff = np.array(
                    [[float(0.23320571), float(-0.11904307), float(0.00389023), float(0.00985417), float(0.55733118)]])


# i = iteration number; M = number of matched features, B = block size
def preemption_function(i, M, B=100):
    return int(np.floor(M * np.power(2, np.floor(-i / B))))


# ported from modules/calib3d/src/five-point.cpp line 373
def scoring_function(p1, p2, essential_mat):
    # x1 = [p1.x, p1.y, 1.0]
    # x2 = [p2.x, p2.y, 1.0]
    # the point is in 2D, so we assume the z parameter as 1
    # the assumption may lea to wrong projections but since we project both points with the essential matrix
    # we don't account for any projection error (the x, y and z will have the same bias in both points)

    # Numpy implementation of transposition and matrix operations is very slow for our computation.
    # I noticed a performance increases from 0.4 fps to 2.4 fps -> 6x

    # E * x1
    Ex1 = [
        essential_mat.item(0) * p1[0] + essential_mat.item(1) * p1[1] + essential_mat.item(2) * 1.0,
        essential_mat.item(3) * p1[0] + essential_mat.item(4) * p1[1] + essential_mat.item(5) * 1.0,
        essential_mat.item(6) * p1[0] + essential_mat.item(7) * p1[1] + essential_mat.item(8) * 1.0
    ]

    # E.transpose() * x1
    Etx2 = [
        essential_mat.item(0) * p2[0] + essential_mat.item(3) * p2[1] + essential_mat.item(6) * 1.0,
        essential_mat.item(1) * p2[0] + essential_mat.item(4) * p2[1] + essential_mat.item(7) * 1.0,
        essential_mat.item(2) * p2[0] + essential_mat.item(5) * p2[1] + essential_mat.item(8) * 1.0
    ]

    # x2.transpose().dot(Ex1)
    x2tEx1 = Ex1[0] * p2[0] + Ex1[1] * p2[1] + Ex1[2] * 1.0
    return float(x2tEx1 * x2tEx1 / (Ex1[0] * Ex1[0] + Ex1[1] * Ex1[1] + Etx2[0] * Etx2[0] + Etx2[1] * Etx2[1]))


def generate_hypotheses(observations, num):
    assert(observations is not None and num > 0)

    hypotheses = []
    while len(hypotheses) < num:
        # get 6 points (5 points to get the essential matrices + one for validation, so to have a sigle 3x3 matrix instead of n 3x3 matrices)
        random_observations = np.random.choice(len(observations), 6, replace=False)

        selected_features = [observations[x] for x in random_observations]

        f1_points = np.array([x[0].pt for x in selected_features])
        f2_points = np.array([x[1].pt for x in selected_features])

        # get the essential matrix/matrices with the 5 point method
        essential_mat = cv2.findEssentialMat(f1_points[0:5], f2_points[0:5], camera_matrix)

        if essential_mat[0] is None:
            continue

        mat = []
        for j in range(0, essential_mat[0].shape[0], 3):
            tmp_mat = np.asmatrix([essential_mat[0][j], essential_mat[0][j + 1], essential_mat[0][j + 2]])
            mat.append(tmp_mat)

        best = scoring_function(f1_points[5], f2_points[5], mat[0])
        best_matrix = mat[0]

        for j in range(1, len(mat)):
            if scoring_function(f1_points[5], f2_points[5], mat[j]) < best:
                best_matrix = mat[j]

        hypotheses.append(best_matrix)

    return hypotheses


def motion_estimation(frame_1, frame_2, matched_features):
    # 1.Randomly permute the observations
    np.random.shuffle(matched_features)

    N = len(matched_features)  # Number of observations
    M = N  # Number of samples to be picked to generate the hypotheses. Set equal to the number of observations
    i = 1  # Preemption iteration
    f = preemption_function(i, M)

    motion_hypotheses = generate_hypotheses(matched_features, f)
    scored_motion_hypotheses = []
    for index, item in enumerate(motion_hypotheses):
        scored_motion_hypotheses.append((index, 0))

    while i <= N and f > 1:
        for h in range(0, len(scored_motion_hypotheses)):
            motion_hypotheses_index = scored_motion_hypotheses[h][0]

            score = scoring_function(matched_features[i-1][0].pt,
                                     matched_features[i-1][1].pt,
                                     motion_hypotheses[motion_hypotheses_index])

            scored_motion_hypotheses[h] = (motion_hypotheses_index, scored_motion_hypotheses[h][1] + score)

        i += 1
        f = preemption_function(i, M)

        scored_motion_hypotheses = sorted(scored_motion_hypotheses, key=lambda x: x[1], reverse=False)[0:f]

    best_hypothesis = motion_hypotheses[scored_motion_hypotheses[0][0]]
    print(best_hypothesis)
    return True



