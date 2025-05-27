import cv2 as cv
import os
import math
import datetime
import numpy as np


def make_video(image_folder, image_files, name, side):
    
    # Read the first image to get the size (height and width)
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv.imread(first_image_path)
    height, width, _ = first_image.shape
    
    if side == "right":
        start_x, end_x = 220, 320
        end_y = 200
        white_threshold = 75
        final_end_x, final_end_y = width-150, 400
    elif side == "left":
        start_x, end_x = 315, 415
        end_y = 200
        white_threshold = 75
        final_end_x, final_end_y = width-30, 400

    # Create a VideoWriter object to save the images as a video
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps = 30  # Frames per second
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{name}_{timestamp}"
    output_video = cv.VideoWriter(output_filename + '.avi', fourcc, fps, (width, height))

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
        
        if raw_frame is None: # Ensure the image was read correctly
            print(f"Could not read image {image_path}")
            continue
        
        # Convert to gray scale
        gray = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)

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
                # cv.imshow('Chessboard Corners', roi)
                # cv.waitKey(0)
                # cv.imshow('Chessboard Corners', gray)
                # cv.waitKey(0)
                found_ball = True
            # else: print("No solid block of white pixels detected.")
            
        if found_ball:
            frame_number += 1
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

                # Draw the bounding box and center on the frame
                cv.rectangle(prev_raw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.circle(prev_raw_frame, center, 5, (0, 0, 255), -1)

                # Print the coordinates of the baseball
                print(f"{frame_number} - Coordinates of baseball: {center}, {prev_center}, Distance: {math.dist(center, prev_center)}")
                prev_center = center
                
                if y > final_end_y or x > final_end_x: break   # End it when the ball is out of range
            
            # Draw the region of interest on the original frame
            top_left = (roi_x_left, roi_y_left)
            bottom_right = (roi_x_end, roi_y_end)
            cv.rectangle(prev_raw_frame, top_left, bottom_right, (0, 0, 255), 2)
            
            if frame_number % 5 == 0:
                cv.imshow(f'Frame {frame_number}', prev_raw_frame)
                cv.waitKey(0)

            output_video.write(prev_raw_frame)  # Write the frame to the video
        
    # Release the VideoWriter object
    output_video.release()
    print("Video has been saved successfully.")
    print(height, width)
    
    
if __name__=="__main__":
    # Load in the image files
    image_folder_L = "../Videos/Video3/L"
    left_images = sorted(os.listdir(image_folder_L))
    left_images = [image for image in left_images if image.endswith(".png")]
    image_folder_R = "../Videos/Video3/R"
    right_images = sorted(os.listdir(image_folder_R))
    right_images = [image for image in right_images if image.endswith(".png")]
    
    make_video(image_folder_L, left_images, "left_throw", "left")
    make_video(image_folder_R, right_images, "left_throw", "right")
    
    
    
# For Right images:
    # 220 <= x <= 320,  y < 200
# For Left images:
    # 315 <= x <= 415,  y < 200
        
# To draw lines (and find the curtain)
# first_image_path = os.path.join(image_folder, image_files[0])
# frame = cv.imread(first_image_path)

# x = 315
# start_point = (x, 0)  # Top of the image
# end_point = (x, 640 - 1)  # Bottom of the image
# x2 = 415 
# start_point2 = (x2, 0)  # Top of the image
# end_point2 = (x2, 640 - 1)  # Bottom of the image
# y = 200  # Middle of the image
# start_pointy = (0, y)  # Left side of the image
# end_pointy = (480 - 1, y)  # Right side of the image

# # Draw the vertical line
# cv.line(frame, start_point, end_point, (0, 0, 255), 2)
# cv.line(frame, start_point2, end_point2, (0, 0, 255), 2)
# cv.line(frame, start_pointy, end_pointy, (0, 0, 255), 2)

# cv.imshow('Chessboard Corners', frame)
# cv.waitKey(0)