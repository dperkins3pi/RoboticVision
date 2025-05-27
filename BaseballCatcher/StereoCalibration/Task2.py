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

        # Display first image if desired
        if i == 0 and display_corners:
            image_with_corners = cv.drawChessboardCorners(color_images[i], (10, 7), corners_refined, ret)
            # Display the image with the corners drawn
            cv.imshow('Chessboard Corners', image_with_corners)
            cv.waitKey(0)
            
    return object_points, image_points

# Load in the saved parameters
left_parameters = np.load('Left Camera.npy', allow_pickle=True).item()
right_parameters = np.load('Right Camera.npy', allow_pickle=True).item()
practice_left_parameters = np.load('Practice Left Camera.npy', allow_pickle=True).item()
practice_right_parameters = np.load('Practice Right Camera.npy', allow_pickle=True).item()
left_mtx = left_parameters['mtx']
left_dist = left_parameters['dist']
right_mtx = right_parameters['mtx']
right_dist = right_parameters['dist']
practice_left_mtx = practice_left_parameters['mtx']
practice_left_dist = practice_left_parameters['dist']
practice_right_mtx = practice_right_parameters['mtx']
practice_right_dist = practice_right_parameters['dist']

left_files, left_color_images, left_gray_images = load_data("StereoSet/L")
right_files, right_color_images, right_gray_images = load_data("StereoSet/R")
practice_left_files, practice_left_color_images, practice_left_gray_images = load_data("Practice/SL")
practice_right_files, practice_right_color_images, practice_right_gray_images = load_data("Practice/SR")

# Get points from chess boards
object_points, left_image_points = get_corners(left_files, left_color_images, left_gray_images)
object_points, right_image_points = get_corners(right_files, right_color_images, right_gray_images)
object_points = [points * 3.88 for points in object_points]  # Chess board size is 3.88" x 3.88"
practice_object_points, practice_left_image_points = get_corners(practice_left_files, practice_left_color_images, practice_left_gray_images)
practice_object_points, practice_right_image_points = get_corners(practice_right_files, practice_right_color_images, practice_right_gray_images)
practice_object_points = [points * 2 for points in practice_object_points]  # Chess board size is 2" x 2"

# Stereo calibration
print(len(object_points), object_points[0].shape)
print(len(left_image_points), left_image_points[0].shape)
print(len(right_image_points), right_image_points[0].shape)
assert 1==0
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv.stereoCalibrate(
    objectPoints=object_points,
    imagePoints1=left_image_points,
    imagePoints2=right_image_points,
    cameraMatrix1=left_mtx,
    distCoeffs1=left_dist,
    cameraMatrix2=right_mtx,
    distCoeffs2=right_dist,
    imageSize=IMAGE_SIZE,
    flags=cv.CALIB_FIX_INTRINSIC
)

# Stereo calibration
practice_ret, practice_mtxL, practice_distL, practice_mtxR, practice_distR, \
    practice_R, practice_T, practice_E, practice_F = cv.stereoCalibrate(
        objectPoints=practice_object_points,
        imagePoints1=practice_left_image_points,
        imagePoints2=practice_right_image_points,
        cameraMatrix1=practice_left_mtx,
        distCoeffs1=practice_left_dist,
        cameraMatrix2=practice_right_mtx,
        distCoeffs2=practice_right_dist,
        imageSize=IMAGE_SIZE,
        flags=cv.CALIB_FIX_INTRINSIC
)
    
# Save parameters into a file
flea_parameters = {'ret': ret, 
        'mtxL': mtxL,
        'distL': distL,
        'mtxR': mtxR,
        'distR': distR,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        }
np.save('Flea_Stereo.npy', flea_parameters)
practice_parameters = {'ret': practice_ret, 
        'mtxL': practice_mtxL,
        'distL': practice_distL,
        'mtxR': practice_mtxR,
        'distR': practice_distR,
        'R': practice_R,
        'T': practice_T,
        'E': practice_E,
        'F': practice_F,
        }
np.save('Practice_Stereo.npy', practice_parameters)

# Print the results
print(f"Rotation Matrix:\n{R}")
print(f"Rotation Vector:\n{cv.Rodrigues(R)[0]}")
print(f"Translation Vector:\n{T}")
print(f"Essential Matrix:\n{E}")
print(f"Fundamental Matrix:\n{F}")
print("\n------Practice-------")
print(f"Rotation Matrix:\n{practice_R}")
print(f"Rotation Vector:\n{cv.Rodrigues(practice_R)[0]}")
print(f"Translation Vector:\n{practice_T}")
print(f"Essential Matrix:\n{practice_E}")
print(f"Fundamental Matrix:\n{practice_F}")