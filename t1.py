import glob

import numpy as np
import cv2
import yaml

boardsize = [6, 7]
numEdgeX = 7
numEdgeY = 6
img_shape = (640,480)
criteria = (cv2.TermCriteria_EPS +
                    cv2.TermCriteria_MAX_ITER, 30, 0.001)
criteria_cal = (cv2.TermCriteria_EPS +
                    cv2.TermCriteria_MAX_ITER, 30, 1e-5)
objp = np.zeros((numEdgeX*numEdgeY, 3), np.float32)
objp[:, :2] = np.mgrid[0:numEdgeX, 0:numEdgeY].T.reshape(-1, 2)

objpoints = []     # 3d points in real world space
imgpoints_l = []   # 2d points in image plane for calibration
imgpoints_r = []
images_right = []
images_left = []

def calibrator(add,imgpoints,b):

    images = glob.glob(add)
    objpoint = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #
        ret, corners = cv2.findChessboardCorners(gray, (boardsize[1], boardsize[0]), None)
        if ret == True:
            if(b):
                images_left.append(img)
            else:
                images_right.append(img)
            objpoint.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    print(np.shape(objpoint))
    print(np.shape( imgpoints))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoint, imgpoints, gray.shape[::-1], None, None)
    for i in objpoint:
        objpoints.append(i)
    return ret, mtx, dist, rvecs, tvecs,objpoint

rt, M1, d1, r1, t1 , objpoints_l = calibrator('data_stereo/left*.jpg',imgpoints_l,1)
# calibrate right camera
rt, M2, d2, r2, t2 , objpoints_r = calibrator('data_stereo/right*.jpg',imgpoints_r,0)

flags = (cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6)

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                    cv2.TERM_CRITERIA_EPS, 100, 1e-5)

#flags = 0
#flags = cv2.CALIB_USE_INTRINSIC_GUESS
#flags = cv2.CALIB_FIX_PRINCIPAL_POINT
#flags = cv2.CALIB_FIX_ASPECT_RATIO
#flags = cv2.CALIB_ZERO_TANGENT_DIST
#flags = cv2.CALIB_FIX_INTRINSIC
#flags = cv2.CALIB_FIX_FOCAL_LENGTH
#flags = cv2.CALIB_FIX_K1...6
#flags = cv2.CALIB_RATIONAL_MODEL
#flags = cv2.CALIB_THIN_PRISM_MODEL
#flags = cv2.CALIB_SAME_FOCAL_LENGTH
#flags = cv2.CALIB_FIX_S1_S2_S3_S4

flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_FOCAL_LENGTH |
         cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 |
         cv2.CALIB_FIX_K6)

T = np.zeros((3, 1), dtype=np.float64)
R = np.eye(3, dtype=np.float64)

ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpoints_l, imgpoints_l,imgpoints_r,
        M1, d1, M2,d2, img_shape, None,None,None,None,
        criteria = stereocalib_criteria,
        flags=flags)
data = {'camera_matrix1': np.asarray(M1).tolist(), 'dist_coeff1': np.asarray(d1).tolist()
    ,'camera_matrix2': np.asarray(M2).tolist(), 'dist_coeff2': np.asarray(d2).tolist()}
with open("stereo_calibration_chess.yaml", "w") as f:
    yaml.dump(data, f)


newCamMtx1, roi1 = cv2.getOptimalNewCameraMatrix(M1, d1,img_shape , 0, img_shape )
newCamMtx2, roi2 = cv2.getOptimalNewCameraMatrix(M2, d2, img_shape, 0, img_shape)


# rectification and undistortion maps which can be used directly to correct the stereo pair
(rectification_l, rectification_r, projection_l,
    projection_r, disparityToDepthMap, ROI_l, ROI_r) = cv2.stereoRectify(
        M1, d1, M2, d2, img_shape, R, T,
        None, None, None, None, None,
        #cv2.CALIB_ZERO_DISPARITY,                  # principal points of each camera have the same pixel coordinates in rect views
        alpha=0)                                   # alpha=1 no pixels lost, alpha=0 pixels lost

leftMapX, leftMapY = cv2.initUndistortRectifyMap(
    M1, d1, rectification_l, projection_l,
    img_shape, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(
    M2, d2, rectification_r, projection_r,
    img_shape, cv2.CV_32FC1)


# compute rectification uncalibrated
### REMAPPING ###
# load images and convert to cv2 format
img_l = images_left[0]
img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
#img_l_undis = cv2.undistort(img_l, M1, d1, None, newCamMtx1)
img_r = images_right[0]
img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
#img_r_undis = cv2.undistort(img_r, M2, d2, None, newCamMtx2)

# remap
imglCalRect = cv2.remap(img_l, leftMapX, leftMapY, cv2.INTER_LINEAR)
imgrCalRect = cv2.remap(img_r, rightMapX, rightMapY, cv2.INTER_LINEAR)

cv2.imshow("calibRectl",imglCalRect)
cv2.imshow("calibRectr",imgrCalRect)
cv2.waitKey()
