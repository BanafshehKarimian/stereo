import numpy as np
import cv2
import glob
import yaml

ret = None
mtx= None
dist= None
rvecs= None
tvecs = None

def test(testfile):
    img = cv2.imread(testfile)
    h, w = img.shape[:2]
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




boardsize = [6,9]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((boardsize[0]*boardsize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:boardsize[1],0:boardsize[0]].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('data/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #
    ret, corners = cv2.findChessboardCorners(gray, (boardsize[1],boardsize[0]),None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
#
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print(ret, mtx, dist)
cv2.destroyAllWindows()
data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

with open("calibration_chess.yaml", "w") as f:
    yaml.dump(data, f)

with open('stereo_calibration_chess.yaml') as f:
    loadeddict = yaml.load(f)
mtx = loadeddict.get('camera_matrix1')
dist = loadeddict.get('dist_coeff1')



images = glob.glob('test/*.jpg')
for fname in images:
    test(fname)



tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error
print ("total error: ", tot_error/len(objpoints))
