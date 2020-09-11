import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import cv2
import os

nx = 9
ny = 6
pattern_points = np.zeros((nx*ny,3), np.float32)
pattern_points[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) 

object_points = []
image_points = []

imgFiles = os.listdir("camera_cal/")

for file in imgFiles:
    print("Processing ",file)
    image = mpimg.imread('camera_cal/'+file)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    # print(len(corners))
    if found:
        object_points.append(pattern_points)
        image_points.append(corners)

        # cv2.drawChessboardCorners(gray, (nx,ny), corners, found)
        # cv2.imshow('corners', image)
        # cv2.waitKey(0)


# Test undistortion on an image
img = mpimg.imread('camera_cal/calibration5.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size,None,None)


dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('camera_cal_output/calibration1_undist.jpg',dst)

# Save the camera calibration result for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_cal_output/mtx_dist_pickle.p", "wb" ) )

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)

plt.show()