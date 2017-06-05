import cv2
import numpy as np
import motion_estimation
from random import randint

t_ref = None
r_ref = None

trajectory = np.zeros((700, 700, 3), dtype=np.uint8)

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
            t_ref = t_ref + (r_ref.dot(t))
            r_ref = r.dot(r_ref)

            print("X:", t_ref[0], " Y:", t_ref[1], " Z:", t_ref[2])
            cv2.circle(trajectory, (t_ref[1]+300, t_ref[0]+300), 1, (0, 255, 0), 1)
            cv2.rectangle(trajectory, (10, 20), (600, 60), (0, 0, 0), -1)
            cv2.imshow("Trajectory", trajectory)
