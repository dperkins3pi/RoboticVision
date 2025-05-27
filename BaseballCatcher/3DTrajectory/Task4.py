import cv2 as cv
import os
import math
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)


def get_points(image_folder, image_files, side):
    
    # Read the first image to get the size (height and width)
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv.imread(first_image_path)
    height, width, _ = first_image.shape
    points = []
    
    if side == "right":
        start_x, end_x = 220, 320
        end_y = 200
        white_threshold = 75
        final_end_x, final_end_y = width-30, 400
    elif side == "left":
        start_x, end_x = 315, 415
        end_y = 200
        white_threshold = 75
        final_end_x, final_end_y = width-30, 400

    prev_center = None
    max_distance = 65   # Maximum allowable distance between consecutive centers
    
    # Loop through the images in the folder and add them to the video
    found_ball = False
    frame_number = 0
    
    for prev_image_file, image_file in zip(image_files[1:], image_files[:-1]):
        image_path = os.path.join(image_folder, image_file)
        raw_frame = cv.imread(image_path)   # Read the image
        prev_image_path = os.path.join(image_folder, prev_image_file)
        prev_raw_frame = cv.imread(prev_image_path)   # Read the image
        frame_number += 1
        
        if raw_frame is None: # Ensure the image was read correctly
            print(f"Could not read image {image_path}")
            continue

        gray = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)  # Convert to gray scale

        if not found_ball:   # Don't do anything until the ball is found
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
                # cv.drawContours(roi, [contour], -1, (255, 255, 0), 2)
                # print(f"Block of white pixels detected")

            if block_detected:
                found_ball = True
            # else: print("No solid block of white pixels detected.")
            
        if found_ball:
            # Get the region of interest  
            roi_x_left, roi_y_left = prev_x-10, prev_y-10
            roi_x_end, roi_y_end = final_end_x, final_end_y
            prev_gray_roi = cv.cvtColor(prev_raw_frame, cv.COLOR_BGR2GRAY)[roi_y_left:roi_y_end, roi_x_left:roi_x_end]
            gray_roi = gray[roi_y_left:roi_y_end, roi_x_left:roi_x_end]
            
            # Calculate the absolute difference
            frame_diff = cv.absdiff(prev_gray_roi, gray_roi)
            
            # Threshold it
            _, thresh = cv.threshold(frame_diff, 25, 100, cv.THRESH_BINARY)
            
            # Get the contours
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            valid_contours = []
            if contours:
                for contour in contours:
                    x, y, w, h = cv.boundingRect(contour)
                    y += prev_y-10
                    x += prev_x-10
                    if x <= start_x: continue   # The ball can't be left of the starting point
                    center = (x + w // 2, y + h // 2)
                    
                    # Check the region is close enough to the last one
                    if (prev_center is None or math.dist(center, prev_center) <= max_distance): valid_contours.append(contour)
            # valid_contours = contours  
            if valid_contours:
                largest_contour = max(valid_contours, key=cv.contourArea)
                x, y, w, h = cv.boundingRect(largest_contour)
                y += prev_y-10
                x += prev_x-10
                prev_y = y
                prev_x = x
                center = (x + w // 2, y + h // 2) 
                
                # print("mgkhj", [frame_number, center[0], center[1]], prev_x, prev_y)
                points.append([frame_number, center[0], center[1]])

                # Draw the bounding box and center on the frame
                cv.rectangle(prev_raw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.circle(prev_raw_frame, center, 5, (0, 0, 255), -1)
                prev_center = center
                
                if y > final_end_y or x > final_end_x: break   # End it when the ball is out of range
                # else:
                #     if frame_number % 5 == 0:
                #         cv.imshow(f'{side}: Frame {frame_number}, ({x}, {y})', prev_raw_frame)
                #         cv.waitKey(0)
                
    return np.array(points)

def get_matching_points(image_folder_L, image_folder_R):
    # Load in the image files
    left_images = sorted(os.listdir(image_folder_L))
    left_images = [image for image in left_images if image.endswith(".png")]
    right_images = sorted(os.listdir(image_folder_R))
    right_images = [image for image in right_images if image.endswith(".png")]
    
    left_points = get_points(image_folder_L, left_images, "left")
    right_points = get_points(image_folder_R, right_images, "right")
    
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
    # Load in parameters
    stereo_parameters = np.load('../StereoCalibration/Flea_Stereo.npy', allow_pickle=True).item()
    mtxL = stereo_parameters['mtxL']
    distL = stereo_parameters['distL']
    mtxR = stereo_parameters['mtxR']
    distR = stereo_parameters['distR']
    T = stereo_parameters['T']
    rectification_parameters = np.load('../StereoCalibration/Rectification.npy', allow_pickle=True).item()
    R1 = rectification_parameters['R1']
    R2 = rectification_parameters['R2']
    P1 = rectification_parameters['P1']
    P2 = rectification_parameters['P2']
    
    # Undistort the points
    undistorted_points_left = cv.undistortPoints(left_points[:, 1:].reshape(-1, 1, 2), mtxL, distL, R=R1, P=P1)
    undistorted_points_right = cv.undistortPoints(right_points[:, 1:].reshape(-1, 1, 2), mtxR, distR, R=R2, P=P2)

    # Triangulation (relative to both left and right camera)
    homogeneous_point_3d_left = cv.triangulatePoints(P1, P2, undistorted_points_left, undistorted_points_right)
    point_3d = homogeneous_point_3d_left[:3] / homogeneous_point_3d_left[3]   # Divide by scale factor
    # point_3d[:, 0] += T[0]/2
    # point_3d[:, 1] += T[1]/2
    # point_3d[:, 2] += T[2]/2
    
    combined_array = np.hstack((left_points[:, 0].reshape(-1, 1), point_3d.T))   # Put frame number back in (in case some frames are skipped)
    return combined_array

    
if __name__=="__main__":
    
    image_folder_L = "../Videos/Video1/L"
    image_folder_R = "../Videos/Video1/R"
    
    stereo_parameters = np.load('../StereoCalibration/Flea_Stereo.npy', allow_pickle=True).item()
    T = stereo_parameters['T']
    
    left_points, right_points = get_matching_points(image_folder_L, image_folder_R)
    point_3d = get_3D_points(left_points, right_points)
    point_3d = point_3d[:-3]  # Last few points are off (because the ball is moving too fast)
    
    # print(left_points, right_points)
    t, x, y, z = point_3d.T

    # Generate smooth estimates for the trajectory
    zs = np.linspace(0, max(z), 500)
    poly_y = np.polyfit(z, y, 2)  # Assume quadratic
    poly_x = np.polyfit(z, x, 1)

    # Create polynomial functions from the fitted coefficients
    p_y = np.poly1d(poly_y)
    p_x = np.poly1d(poly_x)
    ys = p_y(zs)
    xs = p_x(zs)
    
    # Plot it
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.xlim(min(zs), max(zs))  # Set x-axis limit
    plt.ylim(min(min(y), min(ys), min(x), min(xs)), max(max(y), max(x), max(ys), max(xs)))
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.scatter(z, y, s=8, label="Observations")
    plt.plot(zs, ys, color="orange", label='Trajectory')
    plt.xlabel("z")
    plt.ylabel("y")
    plt.legend()
    
    plt.subplot(122)
    plt.xlim(min(zs), max(zs))  # Set x-axis limit
    plt.ylim(min(min(y), min(ys), min(x), min(xs)), max(max(y), max(x), max(ys), max(xs)))
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.scatter(z, x, s=8, label="Observations")
    plt.plot(zs, xs, color="orange", label='Trajectory')
    plt.xlabel("z")
    plt.ylabel("x")
    plt.legend()
    
    plt.suptitle("Trajectory and observations of the ball")
    plt.tight_layout()
    plt.show()
    
    print(f"Final x: {xs[0]}")
    print(f"Final y: {ys[0]}")
    print(f"Final x: {(xs[0]+T[0]/2)[0]}")
    print(f"Final y: {(ys[0]+T[1]/2)[0]}")