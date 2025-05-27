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

def draw_horizontal_lines(image, line_spacing=40):
    img_with_lines = image.copy()
    height, width = image.shape[:2]
    
    # Draw horizontal lines
    for y in range(0, height, line_spacing):
        cv.line(img_with_lines, (0, y), (width, y), (0, 255, 0), 1)
    
    return img_with_lines

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

# Stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(mtxL, distL, mtxR, distR, IMAGE_SIZE, R, T, alpha=0)

# Create undistortion and rectification maps
map1_x, map1_y = cv.initUndistortRectifyMap(mtxL, distL, R1, P1, IMAGE_SIZE, cv.CV_32FC1)
map2_x, map2_y = cv.initUndistortRectifyMap(mtxR, distR, R2, P2, IMAGE_SIZE, cv.CV_32FC1)

# Apply the maps to get rectified images
rectified_left = cv.remap(left_image, map1_x, map1_y, cv.INTER_LINEAR)
rectified_right = cv.remap(right_image, map2_x, map2_y, cv.INTER_LINEAR)

# Get the differences
diff_left = cv.absdiff(rectified_left, left_image)
diff_right = cv.absdiff(rectified_right, right_image)

# Draw horizontal lines
rectified_left = draw_horizontal_lines(rectified_left)
rectified_right = draw_horizontal_lines(rectified_right)

# Save parameters into a file
parameters = {'R1': R1, 
        'R2': R2,
        'P1': P1,
        'P2': P2,
        'Q': Q,
        'roi1': roi1,
        'map1_x': map1_x,
        'map1_y': map1_y,
        'map2_x': map2_x,
        'map2_y': map2_y
        }
np.save('Rectification.npy', parameters)

# Show the images    
cv.imshow('Original Left', left_image)
cv.waitKey(0)
cv.imshow('Rectified Left', rectified_left)
cv.waitKey(0)
cv.imshow('Difference Left', diff_left)
cv.waitKey(0)
cv.imshow('Original Right', right_image)
cv.waitKey(0)
cv.imshow('Rectified Right', rectified_right)
cv.waitKey(0)
cv.imshow('Difference Right', diff_right)
cv.waitKey(0)

# Print the results
print(f"Rectification Rotation Matrix:\n{R1}")
print(f"Rotation Vector:\n{cv.Rodrigues(R1)[0]}")
# Print the results
print()
print(f"Rotation Matrix For Right Image:\n{R2}")
print(f"Rotation Vector:\n{cv.Rodrigues(R2)[0]}")