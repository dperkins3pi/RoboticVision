import numpy as np
import cv2 as cv
import os
import re
from matplotlib import pyplot as plt

# The image sequence was taken at 15.25 mm intervals (moving toward the gas can).
mm_per_frame = 15.25

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
out_dir = "Features_Found"
image_files = os.listdir(image_folder)
image_files = sorted(image_files, key=lambda x: int(re.search(r'\d+', x).group()))  # Make the order work
os.makedirs(out_dir, exist_ok=True)

# Get features for the first image
image_path = os.path.join(image_folder, image_files[0])
raw_frame = cv.imread(image_path)
frame = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)
kp1, des1 = sift.detectAndCompute(frame, None)
times_to_impact = []
prev_frame = frame

for frame_number, image_file in enumerate(image_files[1:]):
    image_path = os.path.join(image_folder, image_file)
    raw_frame = cv.imread(image_path)   # Read the image
    raw_frame = cv.imread(image_path)
    frame = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)  # Convert each frame to 8-bit single channel 
    
    # Find the features
    kp2, des2 = sift.detectAndCompute(frame, None)
    
    # Get the best matches
    matches = bf.knnMatch(des1, des2, k=2)
    all_good = []
    for m, n in matches:
        if m.distance < .75*n.distance:
            all_good.append(m)
    all_good.sort(key=lambda x: x.distance)  # Sort matches by distance
    good = all_good[:top_n]  # Keep only the top N matches
    
    # Compute distances to optical center in both frames
    distances_prev = [distance_from_optical_center(kp1[m.queryIdx]) for m in good]
    distances_curr = [distance_from_optical_center(kp2[m.trainIdx]) for m in good]
    
    # Find the expansion rate of each feature
    expansion_rates = [(d2 - d1) / d1 for d1, d2 in zip(distances_prev, distances_curr)]
    
    # Estimate time to impact
    tti_estimates = [d1 / (d2 - d1) if (d2 - d1) != 0 else np.inf for d1, d2 in zip(distances_prev, distances_curr)]
    tti = np.median(tti_estimates)  # Median helps when it is noisy
    
    print(f"Time to Impact at frame {frame_number+1}: {tti}")
    times_to_impact.append(tti)
    
    frame_with_kp = cv.drawKeypoints(frame, kp2, None, color=(0,255,0))
    cv.imwrite(os.path.join(out_dir, f"frame_{frame_number+1}_all_features.jpg"), frame_with_kp)
    match_img = cv.drawMatches(prev_frame, kp1, frame, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(os.path.join(out_dir, f"frame_{frame_number+1}_matched_features.jpg"), match_img)

    prev_frame = frame
    kp1, des1 = kp2, des2


# Fit a linear model (1st-degree polynomial)
times_to_impact = np.array(times_to_impact) * mm_per_frame
all_frames = np.arange(len(times_to_impact))
ts = np.linspace(0, 50, 500)
m, b = np.polyfit(all_frames, times_to_impact, 1)
ys = ts*m+b
final_estimation = np.round(ts[np.where(ys < 0)[0]][0] * mm_per_frame, 3)

# Plot the time to impact
plt.title(f"Distance To Impact Estimation: {final_estimation} mm from the beginning")
plt.ylabel("Estimated Distance To Impact (mm)")
plt.xlabel("Frame Number")
plt.scatter(all_frames, times_to_impact, color="blue")
plt.plot(ts, ys, color="red")
plt.xlim(0, 50)
plt.ylim(0, 50*mm_per_frame)
plt.savefig(os.path.join(out_dir, f"distance_plot.jpg"))
plt.show()