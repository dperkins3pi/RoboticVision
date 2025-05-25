import cv2 as cv
import numpy as np

# Load in images
img1_color = cv.imread('img2_cropped.jpg')
img1  = cv.imread('img2_cropped.jpg', cv.IMREAD_GRAYSCALE)
img2_color = cv.imread('img.jpg')
img2 = cv.imread('img.jpg', cv.IMREAD_GRAYSCALE)

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Choose only the best matches
all_good = []
for m,n in matches:
    if m.distance < .75*n.distance:
        all_good.append([m])
        
all_good.sort(key=lambda x: x[0].distance)  # Sort matches by distance
top_n = 500  # Limit the number of matches to top N
good = all_good[:top_n]  # Keep only the top N matches

# Get the coordinates of the good matches in both images
points1 = np.float32([kp1[m[0].queryIdx].pt for m in all_good])  # Keypoints in img1
points2 = np.float32([kp2[m[0].trainIdx].pt for m in all_good])  # Keypoints in img2

# Find the homography matrix
H, mask = cv.findHomography(points1, points2, cv.RANSAC, 5.0)
h, w = img1.shape
# Define the four corners of the reference image (cropped)
corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
# Apply the homography to the corners of the reference painting
corners_transformed = cv.perspectiveTransform(corners, H)

# Draw the transformed bounding box on the full image (img2)
img_with_box = cv.polylines(img2_color.copy(), [np.int32(corners_transformed)], True, (0, 255, 0), 3, cv.LINE_AA)

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1_color, kp1, img_with_box, kp2, good, 
                        None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the image
cv.imshow('Matched Image', cv.resize(img3, (0, 0), fx=.25, fy=.25))
cv.waitKey(0)