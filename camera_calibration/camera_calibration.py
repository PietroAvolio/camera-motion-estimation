import numpy as np
import cv2

frames = [93, 145, 177, 202, 229, 247, 263, 276, 293, 309, 320]

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for f in frames:#range(0, 388)
    img = cv2.imread("../media/camera_calibration_frames/Frame_" + str(f) + ".jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        #img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        #cv2.imshow('img', img)

    #cv2.waitKey(500)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("RetValue")
print(ret)
print()

print("CameraMatrix")
print(mtx)
print()

print("DistCoeffs")
print(dist)
print()

print("Rvecs")
print(rvecs)
print()

print("tvecs")
print(tvecs)
print()

cv2.destroyAllWindows()