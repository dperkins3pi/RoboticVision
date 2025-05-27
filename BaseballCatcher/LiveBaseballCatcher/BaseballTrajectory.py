import cv2 as cv
import os
import math
from matplotlib import pyplot as plt
import numpy as np

# Load in parameters
# stereo_parameters = np.load('StereoCalibration/Flea_Stereo.npy', allow_pickle=True).item()
# mtxL = stereo_parameters['mtxL']
# distL = stereo_parameters['distL']
# mtxR = stereo_parameters['mtxR']
# distR = stereo_parameters['distR']
# T = stereo_parameters['T']
# rectification_parameters = np.load('StereoCalibration/Rectification.npy', allow_pickle=True).item()
# R1 = rectification_parameters['R1']
# R2 = rectification_parameters['R2']
# P1 = rectification_parameters['P1']
# P2 = rectification_parameters['P2']

mtxL = np.array([[1.68788706e+03, 0.00000000e+00, 3.26074785e+02], 
                [0.00000000e+00, 1.69033873e+03, 2.29100136e+02], 
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distL = np.array([[-4.88594653e-01, -1.24679131e+00, 3.89316951e-03,  7.85455406e-04, 3.05969397e+01]])
mtxR = np.array([[1.69280103e+03, 0.00000000e+00, 3.22865638e+02],
                [0.00000000e+00, 1.69707855e+03, 2.01990927e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distR = np.array([[-5.33564467e-01, 5.23309648e+00, 5.25438881e-03, 3.13336667e-03, -1.15454612e+02]])
T = np.array([[-20.38998479],[-0.02986332], [0.61417699]])
R1 = np.array([[ 0.9993875, 0.00516153, -0.03461194],
            [-0.00487561, 0.99995333, 0.00834004],
            [0.03465337, -0.00816618, 0.99936603]])
R2 = np.array([[0.99954558, 0.00146394, -0.03010782],
            [-0.00171244, 0.99996467, -0.00822953],
            [0.0300947, 0.00827735, 0.99951278]])
P1 = np.array([[1.69377887e+03, 0.00000000e+00, 3.82358589e+02, 0.00000000e+00],
    [0.00000000e+00, 1.69377887e+03, 2.14102676e+02, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])
P2 = np.array([[1.69377887e+03, 0.00000000e+00, 3.82358589e+02, -3.45518263e+04],
    [ 0.00000000e+00, 1.69377887e+03, 2.14102676e+02, 0.00000000e+00],
    [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])


def find_ball(raw_frame, side):
    # Find when the ball is in frame
    prev_x, prev_y, prev_center = None, None, None

    if side == "right":
        start_x, end_x = 220, 320
        # left_end1
        end_y = 200
        white_threshold = 75
    elif side == "left":
        start_x, end_x = 315, 415
        end_y = 200
        white_threshold = 75
    
    gray = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)  # Convert to gray scale

    # Extract the ROI (Region of Interest) and get contours
    roi = gray[:end_y, start_x:end_x]  # Crop the image 
    _, binary = cv.threshold(roi, white_threshold, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    block_detected = False
    for contour in contours:  # If contours is not empty, a block is detected
        block_detected = True
        
        x, y, w, h = cv.boundingRect(contour)
        prev_x, prev_y = x, y
        prev_center = (x + w // 2 + start_x, y + h // 2)

    if prev_x is not None: prev_x = prev_x + start_x  # Asjust to account for roi

    return block_detected, prev_x, prev_y, prev_center


def get_points(prev_raw_frame, raw_frame, side, frame_number, prev_x, prev_y, prev_center, x_begining, y_begining, draw_box=True, first_frame=False):
    CONTOUR_METHOD = True
    height, width, _ = raw_frame.shape
    point, done = None, False
    
    if side == "right":
        start_x, end_x = 220, 320
        final_end_x, final_end_y = width-30, 400
        # stop_y = 350
        stop_y = 450
    elif side == "left":
        start_x, end_x = 315, 415
        final_end_x, final_end_y = width-30, 400
        # stop_y = 250
        stop_y = 350
    max_distance = 65   # Maximum allowable distance between consecutive centers

    gray = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)  # Convert to gray scale
    
    # Get the region of interest
    roi_x_left, roi_y_top = prev_x-10, prev_y-20
    roi_x_right, roi_y_bottom = min(roi_x_left+70, final_end_x), min(roi_y_top+85, final_end_y)
    # roi_x_right, roi_y_bottom = final_end_x, final_end_y
    
    gray = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)  # Convert to gray scale
    prev_gray_roi = cv.cvtColor(prev_raw_frame, cv.COLOR_BGR2GRAY)[roi_y_top:roi_y_bottom, roi_x_left:roi_x_right]
    gray_roi = gray[roi_y_top:roi_y_bottom, roi_x_left:roi_x_right]
    
    # Calculate the absolute difference
    frame_diff = cv.absdiff(prev_gray_roi, gray_roi)
    
    # Threshold it
    _, thresh = cv.threshold(frame_diff, 30, 255, cv.THRESH_BINARY)
    
    if CONTOUR_METHOD:
        # Get the contours
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        if contours:
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                x += roi_x_left
                y += roi_y_top
                # if x <= start_x: continue   # The ball can't be left of the starting point
                # print(x, y, prev_x, prev_y)
                if side=="left" and prev_y < 90 and prev_x < 380 and y - prev_y > 5: continue
                if side=="left" and x > 580: continue
                if x - prev_x < -15: continue
                if prev_y < 80 and y - prev_y > 20: continue
                elif prev_y < 120 and y - prev_y > 30: continue
                elif prev_y < 160 and y - prev_y > 50: continue
                center = (x + w // 2, y + h // 2)
                
                # Check the region is close enough to the last one
                if (prev_center is None or np.linalg.norm(np.array(center) - np.array(prev_center)) <= max_distance): valid_contours.append(contour)
        
        # valid_contours = contours  
        if valid_contours:
            largest_contour = max(valid_contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest_contour)
            x += roi_x_left
            y += roi_y_top
            prev_y = y
            prev_x = x
            center = (x + w // 2, y + h // 2) 
            
            point = np.array([frame_number, center[0], center[1]])

            # Draw the bounding box and center on the frame
            if draw_box:
                if not first_frame:
                    cv.rectangle(prev_raw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.circle(prev_raw_frame, center, 3, (0, 0, 255), -1)
                else:
                    cv.rectangle(raw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.circle(raw_frame, center, 3, (0, 0, 255), -1)
                
            prev_center = center
            
            if point[1] > stop_y or point[0] > final_end_x:
                done = True   # End it when the ball is out of range
            # else:
            #     if frame_number % 5 == 0:
            #         cv.imshow(f'{side}: Frame {frame_number}, ({x}, {y})', prev_raw_frame)
            #         cv.waitKey(0)
    else:   
        mask = (gray_roi >= 35) & (gray_roi <= 100)
        try: 
            thresh[~mask] = 0
            white_pixels = np.argwhere(thresh > 0)  # Returns (row, col) = (y, x)
            # Convert thresh to color image
            color_thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)

            # print(side, x_begining, prev_x, abs(x_begining - prev_x))
            if len(white_pixels) > 25:
                    weight_map = cv.GaussianBlur(thresh.astype(np.float32), (5, 5), 0)
                    weights = weight_map[white_pixels[:, 0], white_pixels[:, 1]]

                    # Compute weighted mean for better stability
                    cy, cx = np.average(white_pixels, axis=0, weights=weights).astype(int)
                    # Compute mean of y and x coordinates
                    # print("number of pixels", len(white_pixels))
                    # cy, cx = np.mean(white_pixels, axis=0).astype(int)  # Ensure correct order
                    # cy, cx = np.average(white_pixels, axis=0, weights=weights).astype(int)
                    
                    # Draw the center
                    cx += roi_x_left
                    cy += roi_y_top
                    
                    # if side=="left" and prev_x > 400 and len(white_pixels) < 100:
                    #     point = None
                    
                    # elif side=="left" and cy > 140 and len(white_pixels) < 200: point = None
                    if False: pass
                    else:
                        cv.circle(color_thresh, (cx-roi_x_left, cy-roi_y_top), 3, (0, 0, 255), -1)  # Red dot at center
                        prev_y = cy - 20
                        prev_x = cx - 20
                        # print(f"{side}, {frame_number}, Center of the white pixels: ({cx}, {cy})")
                        cv.circle(raw_frame, (cx, cy), 3, (0, 0, 255), -1)  # Red dot at center
                        point = np.array([frame_number, cx, cy])
            
            # block_size = 4 
            # try: print("pixels\n", gray[cy-block_size: cy+block_size+1, cx-block_size : cx+block_size+1])
            # except: pass
        except: pass

    if not first_frame: return point, prev_x, prev_y, prev_center, prev_raw_frame, done
    else: return point, prev_x, prev_y, prev_center, raw_frame, done


def get_matching_points(left_points, right_points):
    # print(left_points, right_points)
    left_frames, right_frames = left_points[:, 0], right_points[:, 0]
    
    # Find all the matches
    matches = np.intersect1d(left_frames, right_frames)
    
    # Get all the points that match
    matched_left_points, matched_right_points = [], []
    for the_match in matches:
        idx1 = np.where(left_frames == the_match)[0][0]
        idx2 = np.where(right_frames == the_match)[0][0]
        matched_left_points.append(left_points[idx1])
        matched_right_points.append(right_points[idx2])
        
    left_points = np.array(matched_left_points)
    right_points = np.array(matched_right_points)
    
    left_points[:,0] -= left_points[0,0]
    right_points[:,0] -= right_points[0,0]
    
    return left_points.astype(np.float32), right_points.astype(np.float32)

def get_3D_points(left_points, right_points):
    # Undistort the points
    undistorted_points_left = cv.undistortPoints(left_points[:,1:].reshape(-1, 1, 2), mtxL, distL, R=R1, P=P1)
    undistorted_points_right = cv.undistortPoints(right_points[:,1:].reshape(-1, 1, 2), mtxR, distR, R=R2, P=P2)

    # Triangulation (relative to both left and right camera)
    homogeneous_point_3d_left = cv.triangulatePoints(P1, P2, undistorted_points_left, undistorted_points_right)
    point_3d = homogeneous_point_3d_left[:3] / homogeneous_point_3d_left[3]   # Divide by scale factor
    
    z_coords = point_3d[2, :]
    last_valid_index = len(z_coords) - 1

    # Clean out noisy end
    # Traverse the array from the end to the beginning
    for i in range(len(z_coords) - 1, 0, -1):
        if z_coords[i] < z_coords[i - 1]:
            last_valid_index = i
            break
    point_3d = point_3d[:, :last_valid_index-0]
    
    # print(point_3d, np.shape(point_3d))
    # point_3d[:, 0] += T[0]/2
    # point_3d[:, 1] += T[1]/2
    # point_3d[:, 2] += T[2]/2
    
    combined_array = np.hstack((left_points[:last_valid_index-0, 0].reshape(-1, 1), point_3d.T))   # Put frame number back in (in case some frames are skipped)
    return combined_array

def get_3D_points(left_points, right_points):
    # Undistort the points
    undistorted_points_left = cv.undistortPoints(left_points[:,1:].reshape(-1, 1, 2), mtxL, distL, R=R1, P=P1)
    undistorted_points_right = cv.undistortPoints(right_points[:,1:].reshape(-1, 1, 2), mtxR, distR, R=R2, P=P2)

    # Triangulation (relative to both left and right camera)
    homogeneous_point_3d_left = cv.triangulatePoints(P1, P2, undistorted_points_left, undistorted_points_right)
    point_3d = homogeneous_point_3d_left[:3] / homogeneous_point_3d_left[3]   # Divide by scale factor
    
    z_coords = point_3d[2, :]
    last_valid_index = len(z_coords) - 1

    # Clean out noisy end
    # Traverse the array from the end to the beginning
    for i in range(len(z_coords) - 1, 0, -1):
        if z_coords[i] < z_coords[i - 1]:
            last_valid_index = i
            break
    point_3d = point_3d[:, :last_valid_index-0]
    
    # print(point_3d, np.shape(point_3d))
    # point_3d[:, 0] += T[0]/2
    # point_3d[:, 1] += T[1]/2
    # point_3d[:, 2] += T[2]/2
    
    combined_array = np.hstack((left_points[:last_valid_index-0, 0].reshape(-1, 1), point_3d.T))   # Put frame number back in (in case some frames are skipped)
    return combined_array

def get_trajectory(point_3d, z_offset):
    # print(point_3d)
    t, x, y, z = point_3d.T
    z = z - z_offset
    
    # Generate smooth estimates for the trajectory
    zs = np.linspace(0, max(z), 500)
    poly_x = np.polyfit(z, x, 1)
    poly_y = np.polyfit(z, y, 2)  # Assume quadratic

    # Create polynomial functions from the fitted coefficients
    p_x = np.poly1d(poly_x)
    p_y = np.poly1d(poly_y)
    xs = p_x(zs)
    ys = p_y(zs)
    
    # Clean the interpolation by removing the 2 most noisy points
    # errors_x = np.abs(x - p_x(z))  # Absolute error for x
    # errors_y = np.abs(y - p_y(z))  # Absolute error for y
    # total_errors = errors_x + errors_y
    # noisy_indices = np.argsort(total_errors)[-2:]  # Get indices of the largest errors
    # z_cleaned = np.delete(z, noisy_indices)  # Remove noisy z values
    # x_cleaned = np.delete(x, noisy_indices)  # Remove corresponding x values
    # y_cleaned = np.delete(y, noisy_indices)  # Remove corresponding y values

    # #Refit polynomial to the cleaned data
    # poly_x_cleaned = np.polyfit(z_cleaned, x_cleaned, 1)
    # poly_y_cleaned = np.polyfit(z_cleaned, y_cleaned, 2)
    
    # p_x_cleaned = np.poly1d(poly_x_cleaned)
    # p_y_cleaned = np.poly1d(poly_y_cleaned)
    
    # xs = p_x_cleaned(zs)
    # ys = p_y_cleaned(zs)
    
    return xs[0], ys[0]
    
    
def plot_trajectory(point_3d, z_offset):
    # print(point_3d)
    t, x, y, z = point_3d.T
    z = z - z_offset
    # Generate smooth estimates for the trajectory
    zs = np.linspace(0, max(z), 500)
    poly_x = np.polyfit(z, x, 1)
    poly_y = np.polyfit(z, y, 2)  # Assume quadratic

    # Create polynomial functions from the fitted coefficients
    p_x = np.poly1d(poly_x)
    p_y = np.poly1d(poly_y)
    xs = p_x(zs)
    ys = p_y(zs)
    
    # # Clean the interpolation by removing the 2 most noisy points
    # errors_x = np.abs(x - p_x(z))  # Absolute error for x
    # errors_y = np.abs(y - p_y(z))  # Absolute error for y
    # total_errors = errors_x + errors_y
    # noisy_indices = np.argsort(total_errors)[-2:]  # Get indices of the largest errors
    # z_cleaned = np.delete(z, noisy_indices)  # Remove noisy z values
    # x_cleaned = np.delete(x, noisy_indices)  # Remove corresponding x values
    # y_cleaned = np.delete(y, noisy_indices)  # Remove corresponding y values

    # #Refit polynomial to the cleaned data
    # poly_x_cleaned = np.polyfit(z_cleaned, x_cleaned, 1)
    # poly_y_cleaned = np.polyfit(z_cleaned, y_cleaned, 2)
    
    # p_x_cleaned = np.poly1d(poly_x_cleaned)
    # p_y_cleaned = np.poly1d(poly_y_cleaned)
    
    # xs = p_x_cleaned(zs)
    # ys = p_y_cleaned(zs)
    z_cleaned, y_cleaned, x_cleaned = z, y, x
    
    # Plot it
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.xlim(min(zs), max(zs))  # Set x-axis limit
    plt.ylim(min(min(y), min(ys), min(x), min(xs)), max(max(y), max(x), max(ys), max(xs)))
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.scatter(z_cleaned, y_cleaned, s=8, label="Observations")
    plt.plot(zs, ys, color="orange", label='Trajectory')
    plt.xlabel("z")
    plt.ylabel("y")
    plt.legend()
    
    plt.subplot(122)
    plt.xlim(min(zs), max(zs))  # Set x-axis limit
    plt.ylim(min(min(y), min(ys), min(x), min(xs)), max(max(y), max(x), max(ys), max(xs)))
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.scatter(z_cleaned, x_cleaned, s=8, label="Observations")
    plt.plot(zs, xs, color="orange", label='Trajectory')
    plt.xlabel("z")
    plt.ylabel("x")
    plt.legend()
    
    plt.suptitle("Trajectory and observations of the ball")
    plt.tight_layout()
    plt.show()
    
    print(f"Final x (left): {xs[0]}")
    print(f"Final y (left): {ys[0]}")
    # print(f"Final x (right): {(xs[0]+T[0]/2)[0]}")
    # print(f"Final y (right): {(ys[0]+T[1]/2)[0]}")