import cv2
import numpy as np
import motion_estimation
from random import randint

t_ref = None
r_ref = None


def get_absolute_scale(matched_features):
    random_point = matched_features[0] #randint(0, len(matched_features)-1)
    previous_point = random_point[0].pt
    current_point = random_point[1].pt

    return np.sqrt((current_point[0]-previous_point[0])*(current_point[0]-previous_point[0]) +
                   (current_point[1] - previous_point[1]) * (current_point[1] - previous_point[1]))


def process_motion_hypothesis(essential_matrix, matched_features):
    global current_coordinates, t_ref, r_ref
    success, r, t, mask = cv2.recoverPose(essential_matrix,
                                          np.array([x[0].pt for x in matched_features]),
                                          np.array([x[1].pt for x in matched_features]),
                                          motion_estimation.camera_matrix)

    if success:
        r = np.asmatrix(r)
        t = np.asmatrix(t)

        if t_ref is None or r_ref is None:
            t_ref = t
            r_ref = r
        else:
            absolute_scale = get_absolute_scale(matched_features)
            print(absolute_scale)

            if absolute_scale > 0.1:
                t_ref = t_ref + absolute_scale*(r_ref.dot(t))
                r_ref = r.dot(r_ref)

            print("X:", t_ref[0], " Y:", t_ref[1], " Z:", t_ref[2])
