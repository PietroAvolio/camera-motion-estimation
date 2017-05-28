import cv2
import numpy as np

previous_frame = None

class Frame:
    def __init__(self, key_points, frame_id):
        self.kp = key_points
        self.fid = frame_id

    def getKeyPoints(self):
        return self.kp

    def getFrameId(self):
        return self.fid

def onNewFeaturesDiscovered(kp):
    nf = Frame(kp, (previous_frame.getFrameId() if previous_frame is not None else 0))

def featuresMatching(f1, f2):
