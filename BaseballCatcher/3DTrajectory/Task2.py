import os
import cv2 as cv
import numpy as np
np.set_printoptions(suppress=True)

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

# Draw circles on the points
def draw_circles(image, points, color):
    if color=="green": color = (0, 255, 0)
    elif color=="red": color = (0, 0, 255)
    for pt in points:
        x, y = pt.ravel()
        cv.circle(image, (int(x), int(y)), radius=3, color=color)

# Load in parameters
stereo_parameters = np.load('../StereoCalibration/Flea_Stereo.npy', allow_pickle=True).item()
mtxL = stereo_parameters['mtxL']
distL = stereo_parameters['distL']
mtxR = stereo_parameters['mtxR']
distR = stereo_parameters['distR']
R = stereo_parameters['R']
T = stereo_parameters['T']
E = stereo_parameters['E']
F = stereo_parameters['F']
rectification_parameters = np.load('../StereoCalibration/Rectification.npy', allow_pickle=True).item()
R1 = rectification_parameters['R1']
R2 = rectification_parameters['R2']
P1 = rectification_parameters['P1']
P2 = rectification_parameters['P2']
Q = rectification_parameters['Q']
roi1 = rectification_parameters['roi1']
map1_x = rectification_parameters['map1_x']
map1_y = rectification_parameters['map1_y']
map2_x = rectification_parameters['map2_x']
map2_y = rectification_parameters['map2_y']

# Load in the data
left_files, left_color_images, left_gray_images = load_data("../StereoCalibration/StereoSet/L")
right_files, right_color_images, right_gray_images = load_data("../StereoCalibration/StereoSet/R")

# Chose 2 corresponding images
left_image_color, right_image_color = left_color_images[0], right_color_images[0]
left_image_gray, right_image_gray = left_gray_images[0], right_gray_images[0]

# Find the chess board corners (Corners outputs a list of all the detected corners)
left_ret, left_corners = cv.findChessboardCorners(left_image_gray, CHESS_BOARD_SIZE, None)
right_ret, right_corners = cv.findChessboardCorners(right_image_gray, CHESS_BOARD_SIZE, None)

# Refine the corner locations using cornerSubPix
left_corners_refined = cv.cornerSubPix(left_image_gray, left_corners, (11, 11), (-1, -1), 
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.1))
right_corners_refined = cv.cornerSubPix(right_image_gray, right_corners, (11, 11), (-1, -1), 
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.1))

# Choose 4 outermost corners
points_left = np.array([left_corners_refined[0], left_corners_refined[9], left_corners_refined[-10], left_corners_refined[-1]])
points_right = np.array([right_corners_refined[0], right_corners_refined[9], right_corners_refined[-10], right_corners_refined[-1]])

# Undistort the points
undistorted_points_left = cv.undistortPoints(points_left, mtxL, distL, R=R1, P=P1)
undistorted_points_right = cv.undistortPoints(points_right, mtxR, distR, R=R2, P=P2)

# Calculate the disparity (which should only be in the x coordinates)
disparity = undistorted_points_left - undistorted_points_right
disparity = disparity[:,:,0]
left_points_with_disparity = np.concatenate((undistorted_points_left, disparity[:, :, np.newaxis]), axis=2)
right_points_with_disparity = np.concatenate((undistorted_points_right, disparity[:, :, np.newaxis]), axis=2)

left_points_3d = cv.perspectiveTransform(left_points_with_disparity, Q)
right_points_3d = cv.perspectiveTransform(right_points_with_disparity, Q)

# Verify that the top left and bottom left point are the correct distance
point_1, point_2 = left_points_3d[0], left_points_3d[2]
print("Left Camera")
print("Distance between top left and bottom left point:", np.linalg.norm(point_1 - point_2))
print("Expected Distance:", 6*3.88)
print("Right Camera")
point_1, point_2 = right_points_3d[0], right_points_3d[2]
print("Distance between top left and bottom left point:", np.linalg.norm(point_1 - point_2))
print("Expected Distance:", 6*3.88)

# Verify that P_l = P_r + [||T||,0,0]^T
print()
print("Left Points:\n", left_points_3d)
print("Right Points:\n", right_points_3d)
Translation = np.zeros_like(right_points_3d)
Translation[:,:,0] = np.linalg.norm(T)
print("Right Points + [||T||,0,0]^T\n", right_points_3d + Translation)
print()

# Triangulation (relative to both left and right camera)
homogeneous_point_3d_left = cv.triangulatePoints(P1, P2, undistorted_points_left, undistorted_points_right)
point_3d_left = homogeneous_point_3d_left[:3] / homogeneous_point_3d_left[3]   # Divide by scale factor

# R_rectified = np.dot(R2.T, np.dot(R, R1))  # Apply rectification to R
T_rectified = np.dot(R2.T, T)  # Apply rectification to T
point_3d_right = point_3d_left + T

print("Previous Method:")
print(f"Left camera coordinates:\n{point_3d_left.T}")
print(f"Right camera coordinates:\n{point_3d_right.T}")