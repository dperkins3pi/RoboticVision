import numpy as np
import cv2 as cv

# Load in the saved parameters
parameters = np.load('task5_parameters.npy', allow_pickle=True).item()
ret = parameters['ret']
mtx = parameters['mtx']
dist = parameters['dist']
rvecs = parameters['rvecs']
tvecs = parameters['tvecs']

# Load in the images
image1 = cv.imread("my_chessboard_photos/photo_1.jpg")
image2 = cv.imread("my_chessboard_photos/photo_14.jpg")
image3 = cv.imread("my_chessboard_photos/photo_22.jpg")

# Undistort the close image and display the difference image
undistorted_1 = cv.undistort(image1, mtx, dist)
diff_1 = cv.absdiff(image1, undistorted_1)
cv.imshow('First Image Difference', diff_1)
cv.waitKey(0)

# Undistort the turned image and display the difference image
undistorted_2 = cv.undistort(image2, mtx, dist)
diff_2 = cv.absdiff(image2, undistorted_2)
cv.imshow('Second Image Difference', diff_2)
cv.waitKey(0)

# Undistort the far image and display the difference image
undistorted_3 = cv.undistort(image3, mtx, dist)
diff_3 = cv.absdiff(image3, undistorted_3)
cv.imshow('Third Image Difference', diff_3)
cv.waitKey(0)

# The code below loads in images taken from the other camera

# # Load in the images
# close = cv.imread("Close.jpg")
# turned = cv.imread("Turned.jpg")
# far = cv.imread("Far.jpg")

# # Undistort the close image and display the difference image
# undistorted_close = cv.undistort(close, mtx, dist)
# diff_close = cv.absdiff(close, undistorted_close)
# cv.imshow('Close Image Difference', diff_close)
# cv.waitKey(0)

# # Undistort the turned image and display the difference image
# undistorted_far = cv.undistort(far, mtx, dist)
# diff_far = cv.absdiff(far, undistorted_far)
# cv.imshow('Far Image Difference', diff_far)
# cv.waitKey(0)

# # Undistort the far image and display the difference image
# undistorted_turned = cv.undistort(turned, mtx, dist)
# diff_turned = cv.absdiff(turned, undistorted_turned)
# cv.imshow('Turned Image Difference', diff_turned)
# cv.waitKey(0)
