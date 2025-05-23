import numpy as np
import cv2 as cv
import os
import re
from matplotlib import pyplot as plt

# The gas can diameter (width) is 59 mm or 2 and 11/16 inches.
gas_can_diameter = 59

# Calibration parameters
mtx = np.array([[825.0900600547, 0.0000000000, 331.6538103208],
                [0.0000000000, 824.2672147458, 252.9284287373],
                [0.0000000000, 0.0000000000, 1.0000000000]])
dist = np.array([[-0.2380769337, 0.0931325835, 0.0003242537, -0.0021901930, 0.4641735616]])
fx, fy, cx, cy = mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]

def distance_from_optical_center(kp):  # Find distance to optical center
    x, y = kp.pt  # Extract keypoint coordinates
    return np.sqrt((x-cx)**2 + (y-cy)**2)  # Euclidean distance

# Define feature detectors and matching algorithm
sift = cv.SIFT_create(nfeatures=2000)
bf = cv.BFMatcher()
top_n = 600  # Limit the number of matches to top N

# Load in the images
image_folder = "Time_To_Impact_Images"
template_file = "roi2.jpg"
out_dir = "Features_Found"
image_files = os.listdir(image_folder)
image_files = sorted(image_files, key=lambda x: int(re.search(r'\d+', x).group()))  # Make the order work
os.makedirs(out_dir, exist_ok=True)

# Get features for the template
raw_frame = cv.imread(template_file)
gas_can = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)
kp0, des0 = sift.detectAndCompute(gas_can, None)
# Define the four corners of the reference image (cropped)
h, w = gas_can.shape
corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
# cv.imshow("gas_can", gas_can)
# cv.waitKey(0)

# Get features for the first image
image_path = os.path.join(image_folder, image_files[0])
raw_frame = cv.imread(image_path)
frame = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)
kp1, des1 = sift.detectAndCompute(frame, None)

def find_width(kp1, des1, undistort=True):
    """Find the width of the can"""
    # Get matches between the template and image
    matches = bf.knnMatch(des0, des1, k=2)
    all_good = []
    for m,n in matches:
        if m.distance < .75*n.distance:
            all_good.append([m]) 
    all_good.sort(key=lambda x: x[0].distance)  # Sort matches by distance
    good = all_good[:top_n]  # Keep only the top N matches

    # Get the coordinates of the good matches in both images
    points1 = np.float32([kp0[m[0].queryIdx].pt for m in good])  # Keypoints in img1
    points2 = np.float32([kp1[m[0].trainIdx].pt for m in good])  # Keypoints in img2
    # Undistort the points
    if undistort:
        points1 = cv.undistortPoints(points1, mtx, dist, None, mtx)
        points2 = cv.undistortPoints(points2, mtx, dist, None, mtx)

    # Find the homography matrix
    H, mask = cv.findHomography(points1, points2, cv.RANSAC, 5.0)
    # Apply the homography to the corners of the reference painting
    corners_transformed = cv.perspectiveTransform(corners, H)
    # Get the width in pixels of the can
    width_pixels = corners_transformed[2,0,0] - corners_transformed[3,0,0]
    
    return width_pixels
    
times_to_impact = []
all_width_pixels = [find_width(kp1, des1)]

for frame_number, image_file in enumerate(image_files[1:]):
    image_path = os.path.join(image_folder, image_file)
    raw_frame = cv.imread(image_path)   # Read the image
    raw_frame = cv.imread(image_path)
    frame = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)  # Convert each frame to 8-bit single channel 
    
    # Find the features
    kp2, des2 = sift.detectAndCompute(frame, None)
    all_width_pixels.append(find_width(kp2, des2))

    kp1, des1 = kp2, des2

# Calculate distance using d=(f*W)/w
distances = (fx * gas_can_diameter) / np.array(all_width_pixels)
all_frames = np.arange(len(distances))
ts = np.linspace(0, 50, 500)
m, b = np.polyfit(all_frames, distances, 1)
ys = ts*m+b
final_estimation = np.round(ys[0],3)

# Plot the time to impact
plt.title(f"Distance To Impact Estimation: {final_estimation} mm from the beginning")
plt.ylabel("Estimated Distance To Impact (mm)")
plt.xlabel("Frame Number")
plt.scatter(all_frames, distances, color="blue")
plt.plot(ts, ys, color="red")
plt.xlim(0, 50)
plt.ylim(0, 650)
plt.savefig(os.path.join(out_dir, f"distance2_plot.jpg"))
plt.show()