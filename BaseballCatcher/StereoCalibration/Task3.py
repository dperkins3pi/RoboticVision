import numpy as np
import cv2 as cv
import os

CHESS_BOARD_SIZE = (10, 7)
IMAGE_SIZE = (640, 480)

# Load in the data
def load_data(image_folder):
    files = os.listdir(image_folder)
    color_images = []  # Lists of (480, 640, 3) color images
    gray_images = []  # Lists of (480, 640) gray images
    for file in files:
        if "png" in file or "bmp" in file:
            image = cv.imread(os.path.join(image_folder, file))
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Get gray scale image
            color_images.append(image)
            gray_images.append(gray_image)
    return files, color_images, gray_images

def get_corners(files, color_images, gray_images, display_corners=False):
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

        print(corners_refined)
        # Display first image if desired
        if i == 0 and display_corners:
            image_with_corners = cv.drawChessboardCorners(color_images[i], (10, 7), corners_refined, ret)
            # Display the image with the corners drawn
            cv.imshow('Chessboard Corners', image_with_corners)
            cv.waitKey(0)
            
    return object_points, image_points

# Draw the epipolar lines and points
def draw_epilines(img, lines, points):
    for line, pt in zip(lines, points):
        a, b, c = line[0]
        x0, y0 = map(int, [0, -c / b])  # Line at x=0
        x1, y1 = map(int, [img.shape[1], -(c + a * img.shape[1]) / b])  # Line at x=max width
        
        # Draw the epipolar line
        cv.line(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
        
        # Draw circles on the points
        x, y = pt.ravel()
        cv.circle(img, (int(x), int(y)), radius=4, color=(0, 255, 0))

# Load in parameters
parameters = np.load('Flea_Stereo.npy', allow_pickle=True).item()
practice_parameters = np.load('Practice_Stereo.npy', allow_pickle=True).item()
mtxL = parameters['mtxL']
distL = parameters['distL']
mtxR = parameters['mtxR']
distR = parameters['distR']
R = parameters['R']
T = parameters['T']
E = parameters['E']
F = parameters['F']

left_files, left_color_images, left_gray_images = load_data("StereoSet/L")
right_files, right_color_images, right_gray_images = load_data("StereoSet/R")

# Chose 2 corresponding images
left_image, right_image = left_color_images[0], right_color_images[0]

# Undistort the images
undistorted_left = cv.undistort(left_image, mtxL, distL)
undistorted_right = cv.undistort(right_image, mtxR, distR)
undistorted_left_gray = cv.cvtColor(undistorted_left, cv.COLOR_BGR2GRAY)
undistorted_right_gray = cv.cvtColor(undistorted_right, cv.COLOR_BGR2GRAY)

# Find the corners
ret, corners_left = cv.findChessboardCorners(undistorted_left_gray, CHESS_BOARD_SIZE, None)  # 10x7 internal corners
if not ret: print("Corners not found")
corners_refined_left = cv.cornerSubPix(undistorted_left_gray, corners_left, (11, 11), (-1, -1), 
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.1))
ret, corners_right = cv.findChessboardCorners(undistorted_right_gray, CHESS_BOARD_SIZE, None)  # 10x7 internal corners
if not ret: print("Corners not found")
corners_refined_right = cv.cornerSubPix(undistorted_right_gray, corners_right, (11, 11), (-1, -1), 
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.1))

# Chose 4 corners and draw circles around them for each image
points_left = np.array([corners_refined_left[0], corners_refined_left[18], corners_refined_left[-27], corners_refined_left[-1]])
points_right = np.array([corners_refined_right[8], corners_refined_right[23], corners_refined_right[46], corners_refined_right[-11]])

# Compute the epipolar lines
lines_right = cv.computeCorrespondEpilines(points_left, 1, F)
lines_left = cv.computeCorrespondEpilines(points_right, 2, F)

# Draw the lines and points
draw_epilines(undistorted_right, lines_right, points_right)
draw_epilines(undistorted_left, lines_left, points_left)

# Show the images    
cv.imshow('Chessboard Corners', undistorted_left)
cv.waitKey(0)
cv.imshow('Chessboard Corners', undistorted_right)
cv.waitKey(0)