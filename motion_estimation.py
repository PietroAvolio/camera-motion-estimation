import cv2
import numpy as np

previous_frame = None
brute_force_matcher = None

camera_matrix = [[1.18848290e+03, 0.00000000e+00, 6.42833462e+02],
                 [0.00000000e+00, 1.18459614e+03, 3.86675542e+02],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

class Frame:
    def __init__(self, image, key_points, key_points_description, frame_id):
        self.kp = key_points
        self.kp_description = key_points_description
        self.fid = frame_id
        self.image = image

    def getKeyPoints(self):
        return self.kp

    def getFrameId(self):
        return self.fid

    def getKeyPointsDescription(self):
        return self.kp_description

    def getImage(self):
        return self.image

def bruteForceMatch(frame_1_kps, frame_1_descriptors, frame_2_kps, frame_2_descriptors):
    global brute_force_matcher

    if brute_force_matcher is None:
        brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = brute_force_matcher.match(frame_1_descriptors, frame_2_descriptors)

    ret = []
    for m in matches:
        ret.append((frame_1_kps[m.queryIdx], frame_2_kps[m.trainIdx]))

    return ret

def onNewFeaturesDiscovered(image, kp, kp_desc):
    global previous_frame
    matched_features = None

    nf = Frame(image, kp, kp_desc, (previous_frame.getFrameId()+1 if previous_frame is not None else 0))

    if previous_frame is not None:
        matched_features = bruteForceMatch(previous_frame.getKeyPoints(), previous_frame.getKeyPointsDescription(), nf.getKeyPoints(), nf.getKeyPointsDescription())
        print("Matched ", len(matched_features), "/", len(kp) ," features btwn frame ", previous_frame.getFrameId(), "-", nf.getFrameId())

        img_copy = nf.getImage().copy()
        cv2.drawKeypoints(img_copy, [x[1] for x in matched_features], img_copy, (0, 255, 0))
        cv2.putText(img_copy, str(len(matched_features)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
        cv2.imshow('matched_features', img_copy)

        np.random.shuffle(matched_features)

    previous_frame = nf
    return matched_features

