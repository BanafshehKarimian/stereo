import numpy as np
import cv2
import glob
import yaml


retval = None
cameraMatrix1 = None
distCoeffs1 = None
cameraMatrix2 = None
distCoeffs2 = None


def test(testfiler,testfilel):
    imgr = cv2.imread(testfiler+".jpg")
    imgl = cv2.imread(testfilel+".jpg")
    h, w = imgl.shape[:2]
    newcameramtxl, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix1, distCoeffs1, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(imgl, cameraMatrix1, distCoeffs1, None, newcameramtxl)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(testfilel+'_result1.png', dst)
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, None, newcameramtxl, (w, h), 5)
    dst = cv2.remap(imgl, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(testfilel+'_result2.png', dst)
    newcameramtxr, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix2, distCoeffs2, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(imgr, cameraMatrix2, distCoeffs2, None, newcameramtxr)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(testfiler+'_result1.png', dst)
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, None, newcameramtxr, (w, h), 5)
    dst = cv2.remap(imgr, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(testfiler+'_result2.png', dst)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
boardsize = [6,9]
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((boardsize[0]*boardsize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:boardsize[1],0:boardsize[0]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = {} # 3d point in real world space
imgpoints = {} # 2d points in image plane.

# calibrate stereo
for side in ['left', 'right']:
    counter = 0
    images = glob.glob('data_stereo/%s*.jpg' %side)
    objpoints[side] = [];
    imgpoints[side] = [];
    find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (boardsize[1],boardsize[0]),find_chessboard_flags)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints[side].append(objp)

            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints[side].append(corners)
            counter += 1

    assert counter == len(images), "missed chessboard!!"


stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
print(np.shape(objpoints['left']))
print(np.shape(imgpoints['left']))
print(np.shape(imgpoints['right']))
stereocalib_flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
retval,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints['left'], imgpoints['left'], imgpoints['right'],None,None,None,None, (640, 480), None,None,None,None,criteria = stereocalib_criteria, flags = stereocalib_flags)
#print(retval,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F)
data = {'camera_matrix1': np.asarray(cameraMatrix1).tolist(), 'dist_coeff1': np.asarray(distCoeffs1).tolist(),'camera_matrix2': np.asarray(cameraMatrix2).tolist(), 'dist_coeff2': np.asarray(distCoeffs2).tolist()}
print(retval,cameraMatrix1, distCoeffs1)
with open("stereo_calibration_chess.yaml", "w") as f:
    yaml.dump(data, f)

#test("test_stereo/left12","test_stereo/right12")

rectify_scale = 0  # 0=full crop, 1=no crop
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2,
                                                  distCoeffs2, (640, 480), R, T,
                                                  alpha=rectify_scale)
left_maps = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (640, 480), cv2.CV_16SC2)
right_maps = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (640, 480), cv2.CV_16SC2)


test_left = cv2.imread('test_stereo/left12.jpg')
test_right = cv2.imread('test_stereo/right12.jpg')
left_img_remap = cv2.remap(test_left, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
right_img_remap = cv2.remap(test_right, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
cv2.imwrite("test_stereo/left12_chess.jpg", test_left)
cv2.imwrite("test_stereo/right12_chess.jpg", test_right)
