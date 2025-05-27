import numpy as np
import cv2 as cv

# Load in the saved parameters
parameters = np.load('task2_parameters.npy', allow_pickle=True).item()
ret = parameters['ret']
mtx = parameters['mtx']
dist = parameters['dist']
rvecs = parameters['rvecs']
tvecs = parameters['tvecs']

# Read in the data
with open('data.txt', 'r') as file:
    lines = file.readlines()

# Find image points
image_points = np.empty((20, 2))
for i in range(20):
    line = lines[i].strip().split()
    point = [float(string) for string in line]
    image_points[i] = point
    
# Find object points
object_points = np.empty((20, 3))
for i in range(20, 40):
    line = lines[i].strip().split()
    point = [float(string) for string in line]
    object_points[i-20] = point

success, rvec, T = cv.solvePnP(object_points, image_points, mtx, dist)
R, _ = cv.Rodrigues(rvec)  # Convert from vector of angles to rotation matrix
# Verify that the rotation matrix is valid
assert np.linalg.norm(R.T @ R - np.eye(3)) < 0.01, "Rotation Matrix Not Orthornormal"

# Print output
print("rvec:\n", rvec)
print("R:\n", R)
print("T:\n", T)