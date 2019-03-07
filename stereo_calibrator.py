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

def test(testfile, mtx, dist):
    img = cv2.imread(testfile+'.jpg')
    h, w = img.shape[:2]
    print(h,w)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(testfile+'_result1.png', dst)
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(testfile+'_result2.png', dst)


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

    #print(np.shape(objpoint))
    #print(np.shape( imgpoints))
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


test('test_stereo/left12',M1,d1)
#print(np.shape(M2))
#print(np.shape(d2))
#print(np.shape(M1))
#print(np.shape(d1))
#print(d1)
#print(d2)
#test('test_stereo/right01',M2,d2)
