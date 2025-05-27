import os
import cv2 as cv
import numpy as np

CHESS_BOARD_SIZE = (10, 7)
IMAGE_SIZE = (640, 480)
image_folder = "images"  # CHANGE THIS BASED ON FILE PATH
display_corners = False   # Change this to true for first task

# Load in the data
files = os.listdir(image_folder)
color_images = []  # Lists of (480, 640, 3) color images
gray_images = []  # Lists of (480, 640) gray images
for file in files:
    if "AR" in file:  # Only load in certain images
        image = cv.imread(os.path.join("images", file))
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Get gray scale image
        color_images.append(image)
        gray_images.append(gray_image)
        
# Get object points array for one arbitrary image
one_object_point = np.zeros((CHESS_BOARD_SIZE[0] * CHESS_BOARD_SIZE[1], 3), np.float32)
one_object_point[:, :2] = np.mgrid[0:CHESS_BOARD_SIZE[0], 0:CHESS_BOARD_SIZE[1]].T.reshape(-1, 2)

# Initialize object and image points
object_points = []
image_points = []

# Find the corners
for i, image in enumerate(gray_images):
    # Find the chess board corners (Corners outputs a list of all the detected corners)
    ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_SIZE, None)  # 10x7 internal corners
    
    # Restart if the corners weren't found
    if not ret:
        print("Corners not found in", files[i])
        continue
    
    # Refine the corner locations using cornerSubPix
    corners_refined = cv.cornerSubPix(image, corners, (11, 11), (-1, -1), 
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    # Output is num_corners x 1 x 2 ((x,y) coordinates at each corner)
    
    # Store object an dimage points
    object_points.append(one_object_point)  # 3D real-world points
    image_points.append(corners_refined.reshape(-1, 2))  # 2D refined image points

    # Display first image if desired
    if i == 0 and display_corners:
        image_with_corners = cv.drawChessboardCorners(color_images[i], (10, 7), corners_refined, ret)
        # Display the image with the corners drawn
        cv.imshow('Chessboard Corners', image_with_corners)
        cv.waitKey(0)

# Calculate the intrinsic and distortion parameters. 
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, IMAGE_SIZE, None, None)
# ret: The overall calibration error, a scalar.
# mtx: The camera matrix K (3x3), containing the intrinsic parameters (focal length, principal point).
# dist: The distortion coefficients (5 parameters: k1, k2, p1, p2, k3).
# rvecs: Rotation vectors for each image, representing the orientation of the camera relative to the calibration pattern.
    # The output here is the angles, not the rotation matrix
# tvecs: Translation vectors for each image, representing the position of the camera relative to the calibration pattern.

# Get the focal length in pixels
fx, fy = mtx[0, 0], mtx[1,1]
# Convert the unit from pixels to mm
fx, fy = fx/135, fy/135    # 135 = 648/4.8

# Print out intrinsic and distortion parameters
print("Focal Length:", fx, "or", fy)
print("Intrinsic Parameters:\n", mtx)
print("Distortion Parameters:", dist)

# Save parameters into a file
parameters = {'ret': ret, 
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs}
np.save('task2_parameters.npy', parameters)
