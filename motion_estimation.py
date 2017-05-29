import cv2
import numpy as np

previous_frame = None
brute_force_matcher = None

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

    if brute_force_matcher == None:
        brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = brute_force_matcher.match(frame_1_descriptors, frame_2_descriptors)

    ret = []
    for m in matches:
        ret.append((frame_1_kps[m.queryIdx], frame_2_kps[m.trainIdx]))

    return ret

def onNewFeaturesDiscovered(image, kp, kp_desc):
    global previous_frame

    nf = Frame(image, kp, kp_desc, (previous_frame.getFrameId()+1 if previous_frame is not None else 0))

    if previous_frame != None:
        matched_features = bruteForceMatch(previous_frame.getKeyPoints(), previous_frame.getKeyPointsDescription(), nf.getKeyPoints(), nf.getKeyPointsDescription())

        print("Matched ", len(matched_features), " features btwn frame ", previous_frame.getFrameId(), "-", nf.getFrameId())

        cv2.drawKeypoints(image, [x[1] for x in matched_features], image, (0, 0, 255))
        cv2.imshow('matched_featres', image)

    previous_frame = nf

