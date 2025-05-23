import cv2 as cv
import os
import math
import numpy as np

def make_video(image_files, name):
    count = 0
    # Read the first image to get the size (height and width)
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv.imread(first_image_path)
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the images as a video
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps = 30  # Frames per second
    output_video = cv.VideoWriter(name + '.avi', fourcc, fps, (width, height))

    prev_center = None
    max_distance = 65   # Maximum allowable distance between consecutive centers
    
    # Loop through the images in the folder and add them to the video
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
        prev_gray = cv.cvtColor(prev_raw_frame, cv.COLOR_BGR2GRAY)

        # Calculate the absolute difference
        frame_diff = cv.absdiff(prev_gray, gray)
        
        # Threshold it
        _, thresh = cv.threshold(frame_diff, 50, 255, cv.THRESH_BINARY)
        
        # Get the contours
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            valid_contours = []
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                
                # Condition: Check if the bounding box is over a white spot
                roi = prev_raw_frame[y:y+h, x:x+w]
                lower_white = np.array([100, 100, 100])  # Light gray/white lower bound
                upper_white = np.array([255, 255, 255])  # Pure white upper bound
                white_mask = cv.inRange(roi, lower_white, upper_white)
                white_pixel_count = cv.countNonZero(white_mask)  # Count the number of white pixels
                area = w * h  # Total area of the bounding box
                
                # Define a threshold for the percentage of white pixels
                white_pixel_ratio = white_pixel_count / area if area > 0 else 0
                white_threshold = 0.2  # Minimum percentage of white pixels required
                
                # Check the distance from the previous center and if the region has enough white pixels
                if (
                    prev_center is None or math.dist(center, prev_center) <= max_distance
                ) and white_pixel_ratio >= white_threshold:
                    valid_contours.append(contour)
            
        if valid_contours:
            largest_contour = max(valid_contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest_contour)
            center = (x + w // 2, y + h // 2)            

            # Draw the bounding box and center on the frame
            cv.rectangle(prev_raw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.circle(prev_raw_frame, center, 5, (0, 0, 255), -1)

            # Print the coordinates of the baseball
            count += 1
            if prev_center is not None: print(f"Coordinates of baseball: {center}, Distance: {math.dist(center, prev_center)}, Num_frams: {count}")
            prev_center = center
        
        # frame = cv.inRange(ray_frame, 50, 250)   # Apply range-based binarization
        frame = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        
        output_video.write(prev_raw_frame)  # Write the frame to the video
        
    # Release the VideoWriter object
    output_video.release()
    print("Video has been saved successfully.")
    
    
if __name__=="__main__":
    # Load in the image files
    image_folder = "Sequence1"
    image_files = sorted(os.listdir(image_folder))
    if not image_files:
        print("No images found in the folder.")
        exit()
    image_files = [image for image in image_files if image.endswith(".png")]
    left_images = [image for image in image_files if image.startswith("L")]
    right_images = [image for image in image_files if image.startswith("R")]
    
    make_video(left_images, "left_throw")
    make_video(right_images, "right_throw")