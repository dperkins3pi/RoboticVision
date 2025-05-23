import cv2 as cv
import numpy as np

# Open the video file
cap = cv.VideoCapture('Video.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties (like frame width, height, and FPS) and Create VideoWriter object
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)
out = cv.VideoWriter('processed_video.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, 
                    (frame_width, frame_height))

# m = 1, 3, 5, 9
m = 5
# win_size = (7, 7), (21, 21) (35, 35), (49, 49)
win_size = (21, 21)
# max_level = 1, 3, 5, 9
max_level = 9

frame_number = 0
num_points = 300
good_olds = [None]*m

# Read the first frame and get features from it
ret, prev_frame = cap.read()
prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
prev_pts = cv.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=10)
prev_pts = np.float32(prev_pts)
all_old_points = []
all_new_points = []

while True:
    
    ret, next_frame = cap.read()  # Read a frame from the video
    if not ret: break
    
    next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
    
    # Calculate optical flow (i.e., track points)
    next_pts, status, error = cv.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None, winSize=win_size, maxLevel=max_level)
    # Select good points (status = 1 indicates a successful match)
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]
    
    if good_olds[frame_number % m] is not None:  # If we actually have an image m frames back
        
        old_pts_5_frames_ago = good_olds[frame_number % m]

        # Find best matches using nearest neighbors (so that we don't have big jumps from changing the order)
        if len(old_pts_5_frames_ago) > 0 and len(good_new) > 0:
            # Compute pairwise distances
            distances = np.linalg.norm(old_pts_5_frames_ago[:, None] - good_new[None, :], axis=2)
            min_indices = np.argmin(distances, axis=1)  # Find closest match for each old point

            # Filter out large jumps (e.g., threshold = 50 pixels)
            valid_pairs = distances[np.arange(len(min_indices)), min_indices] < 50
            selected_old = old_pts_5_frames_ago[valid_pairs]
            selected_new = good_new[min_indices[valid_pairs]]
            
            all_old_points.append(selected_old)
            all_new_points.append(selected_new)

            # Draw motion lines
            for new, old in zip(selected_new, selected_old):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                cv.line(next_frame, (a, b), (c, d), (0, 255, 0), 2)
                cv.circle(next_frame, (a, b), 5, (0, 0, 255), -1)
                
        out.write(next_frame)  # Write the frame to the output video file
        
    # Update the previous frame and points
    good_olds[frame_number % m] = good_old
    prev_gray = next_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)
    
    if len(good_new) < num_points:  # If too many points are lost, find new ones
        prev_pts = cv.goodFeaturesToTrack(next_gray, maxCorners=num_points+100, qualityLevel=0.01, minDistance=10)
        prev_pts = np.float32(prev_pts)
    
    frame_number += 1

cap.release()
out.release()
cv.destroyAllWindows()

# Save parameters into a file
parameters = {'old_points': all_old_points, "new_points": all_new_points}
np.save('matched_points.npy', parameters)