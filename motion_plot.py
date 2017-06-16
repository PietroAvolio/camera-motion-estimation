import cv2
import numpy as np
import motion_estimation
from random import randint


class Trajectory():
    def __init__(self, sid):
        self.sid = sid
        self.t_ref = None
        self.r_ref = None
        self.trajectory = np.zeros((800, 800, 3), dtype=np.uint8)

    def save(self, name):
        cv2.imwrite(name, self.trajectory)

    def process_motion_hypothesis(self, essential_matrix, matched_features):
        success, r, t, mask = cv2.recoverPose(essential_matrix,
                                              np.array([x[0].pt for x in matched_features]),
                                              np.array([x[1].pt for x in matched_features]),
                                              motion_estimation.camera_matrix)

        if success:
            r = np.asmatrix(r)
            t = np.asmatrix(t)

            if self.t_ref is None or self.r_ref is None:
                self.t_ref = t
                self.r_ref = r
            else:
                self.t_ref = self.t_ref + (self.r_ref.dot(t))
                self.r_ref = r.dot(self.r_ref)

            #print("X:", t_ref[0], " Y:", t_ref[1], " Z:", t_ref[2])
            cv2.circle(self.trajectory, (self.t_ref[1]+400, self.t_ref[0]+400), 1, (0, 255, 0), 1)
            cv2.rectangle(self.trajectory, (10, 20), (600, 60), (0, 0, 0), -1)
            cv2.imshow("Trajectory "+str(self.sid), self.trajectory)
