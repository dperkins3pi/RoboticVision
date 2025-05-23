import numpy as np
import cv2 as cv
np.set_printoptions(suppress=True)

# Load in the parameters
parameters = np.load('calibration_parameters.npy', allow_pickle=True).item()
ret = parameters['ret']
mtx = parameters['mtx']
dist = parameters['dist']
rvecs = parameters['rvecs']
tvecs = parameters['tvecs']

# Load in the points
all_points_parameters = np.load('matched_points2.npy', allow_pickle=True).item()
all_old_points = all_points_parameters["old_points"]
all_new_points = all_points_parameters["new_points"]

# Load in the video
cap = cv.VideoCapture('processed_video2_nolines.mp4')
edited_cap = cv.VideoCapture('processed_video2.mp4')
fps = cap.get(cv.CAP_PROP_FPS)   # 30 frames per second

# Undistort the points
all_old_undistorted_points, all_new_undistorted_points = [], []
for points in all_old_points:
    undistorted_points = cv.undistortPoints(points.reshape(-1, 1, 2).astype(np.float32), mtx, dist)
    all_old_undistorted_points.append(undistorted_points.reshape(-1, 2))
for points in all_new_points:
    undistorted_points = cv.undistortPoints(points.reshape(-1, 1, 2).astype(np.float32), mtx, dist)
    all_new_undistorted_points.append(undistorted_points.reshape(-1, 2))
    
    
def find_movement(old_points, new_points):  # Find F, E, R, and T
    # print(np.mean(new_points - old_points, axis=0))
    # Convert to pixel values
    old_pixel_points = cv.convertPointsToHomogeneous(old_points)[:, 0, :]
    old_pixel_points = (mtx @ old_pixel_points.T).T[:, :2]
    new_pixel_points = cv.convertPointsToHomogeneous(new_points)[:, 0, :]
    new_pixel_points = (mtx @ new_pixel_points.T).T[:, :2]
    old_pixel_points, new_pixel_points = old_points, new_points
    
    # Find the fundamental matrix
    F, mask = cv.findFundamentalMat(old_pixel_points, new_pixel_points, method=cv.FM_RANSAC)
    # Find the essential matrix (and normalize it)
    E = mtx.T @ F @ mtx
    U, S, Vt = np.linalg.svd(E)
    E = U @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) @ Vt
    
    # Find the essential matrix sing OpenCV
    E2, mask = cv.findEssentialMat(old_pixel_points, new_pixel_points, mtx, method=cv.RANSAC)

    # Get R and T
    retval, R, T, mask = cv.recoverPose(E, old_pixel_points, new_pixel_points, mtx, mask=None)
    return F, E, E2, R, T


def get_matrices_and_images(frame_number):
    F, E, E2, R, T = find_movement(all_old_points[frame_number], all_new_points[frame_number])

    # # Read the frame
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number-5)
    ret, frame = cap.read()
    if ret:
        # Resize the frame to a smaller size (e.g., 50% of the original)
        scale_percent = 50  # Adjust this to change the size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized_frame = cv.resize(frame, (width, height))

        # Display the resized frame
        cv.imshow(f'Frame {frame_number}', resized_frame)
        cv.waitKey(0)  # Wait until a key is pressed
        cv.destroyAllWindows()
    else:
        print("Failed to retrieve frame.")
        
    # Read the frame
    edited_cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = edited_cap.read()
    if ret:
        # Resize the frame to a smaller size (e.g., 50% of the original)
        scale_percent = 50  # Adjust this to change the size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized_frame = cv.resize(frame, (width, height))

        # Display the resized frame
        cv.imshow(f'Frame {frame_number}', resized_frame)
        cv.waitKey(0)  # Wait until a key is pressed
        cv.destroyAllWindows()
    else:
        print("Failed to retrieve frame.")

    print("F", F)
    print("E", E)
    # print("E2", E2)
    print("R", R)
    print("T", T)
    
# Find frames for each of the 5 movements
    # Zoom in - 0.75 second --> frame 23-5
    # Zoom out - 1.5 second --> frame 45-5
    # Rotate clockwise - 6 second --> frame 180-5
    # Move down - 8 seconds --> frame 240-5
    # Move up -- 10 seconds --> frame 300-5
    # Move right (quite small and breif) -- 11 second --> frame 330-5
zoom_in, zoom_out = 18, 48
rotate = 170
move_down, move_right = 239, 329

get_matrices_and_images(move_down)