import numpy as np
import cv2 as cv

# Load in the saved parameters
parameters = np.load('task2_parameters.npy', allow_pickle=True).item()
ret = parameters['ret']
mtx = parameters['mtx']
dist = parameters['dist']
rvecs = parameters['rvecs']
tvecs = parameters['tvecs']

# Load in the images
close = cv.imread("Close.jpg")
turned = cv.imread("Turned.jpg")
far = cv.imread("Far.jpg")

# Undistort the close image and display the difference image
undistorted_close = cv.undistort(close, mtx, dist)
diff_close = cv.absdiff(close, undistorted_close)
cv.imshow('Close Image Difference', diff_close)
cv.waitKey(0)

# Undistort the turned image and display the difference image
undistorted_far = cv.undistort(far, mtx, dist)
diff_far = cv.absdiff(far, undistorted_far)
cv.imshow('Far Image Difference', diff_far)
cv.waitKey(0)

# Undistort the far image and display the difference image
undistorted_turned = cv.undistort(turned, mtx, dist)
diff_turned = cv.absdiff(turned, undistorted_turned)
cv.imshow('Turned Image Difference', diff_turned)
cv.waitKey(0)
