"""
ECEn-631 Visual Inspection Project
"""

import cv2 as cv
import numpy as np
import os
from sklearn.svm import LinearSVC
'''
Set WEBCAM to 1 to use your webcam or 0 to use the Flea2 cameras on the lab machine
Set CATCHER to 1 to use the catcher connected to the lab machine or 0 to use your own computer
'''
WEBCAM = 1
CATCHER = 0
PRODUCT = 'Bad'
VIDEO_PATH = 'videos/' + PRODUCT + '_Cleaned.mp4'    # TODO: Make it None when doing it live
# VIDEO_PATH = None
OUTPUT_DIR = 'contours/' + PRODUCT
WEIGHTS_PATH = 'svc_model.npz'
SAVE_IMAGES = False   # Make true if you want to save the images


# Initialize a new LinearSVC model and assign it the weights and intercept
loaded_data = np.load(WEIGHTS_PATH)
clf = LinearSVC(random_state=0, tol=1e-5)
clf.coef_ = loaded_data['coef']
clf.intercept_ = loaded_data['intercept']

if VIDEO_PATH is not None:
    camera = cv.VideoCapture(VIDEO_PATH)
else:
    if WEBCAM:
        camera = cv.VideoCapture(0)
    else:
        from src.Flea2Camera2 import FleaCam
        camera = FleaCam()

if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()
    
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
videout = cv.VideoWriter('./Video1.avi', cv.VideoWriter_fourcc(*'XVID'), 25, (width, height))  # Video format
frame_count = 1

while True:
    # Get Opencv Frame
    if not camera.isOpened():
        ret, frame = camera.read()
    else:
        if WEBCAM:
            ret, frame = camera.read()
        else:
            frame = camera.getFrame()
        
    if not ret:  # Break the loop if there are no frames left
        print("End of video.")
        break

    ########### Make edits to the frame #####################
    # Convert frame to grayscale and apply thresholding
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 100, 200, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to determine which ones are valid and draw bounding boxes for large objects
    valid_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if 1000 < area < 1000000:  # Example: Filter small and overly large areas
            x, y, w, h = cv.boundingRect(contour)   # Get the bounding box for the contour
            if 300 < x < 1500 and 75 < h < 800 and 75 < w < 800:
                valid_contours.append((contour, area, x, y, w, h))
                
    # Merge contours that are close to each other
    merged_contours = []
    for i, (contour1, area1, x1, y1, w1, h1) in enumerate(valid_contours):
        keep = True
        for j, (contour2, area2, x2, y2, w2, h2) in enumerate(merged_contours):
            # Check if the bounding boxes are sufficiently close
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx < 500 and dy < 500:  # Define a distance threshold 
                keep = False
                break
        if keep:
            merged_contours.append((contour1, area1, x1, y1, w1, h1))
            
    # Extract and save crops
    if VIDEO_PATH is not None and SAVE_IMAGES:
        for idx, (contour, area, x, y, w, h) in enumerate(valid_contours):
            # Crop the region
            crop = frame[y:y + h, x:x + w]

            # Resize to a fixed size (e.g., 128x128)
            crop_resized = cv.resize(crop, (128, 128))

            # Normalize the pixel values (e.g., scale to [0, 1])
            crop_normalized = crop_resized / 255.0

            # Save the crop for later use
            filename = f"{OUTPUT_DIR}/frame{frame_count}_object{idx}.png"
            cv.imwrite(filename, (crop_normalized * 255).astype(np.uint8))  # Save as an image
                
    # Iterate through contours to draw bounding boxes for large objects
    for contour, area, x, y, w, h in merged_contours:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv.imshow('frame',frame)

    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

    videout.write(frame)   
    
    frame_count += 1

camera.release()
    
# ssh dbp52@ssh.et.byu.edu:~/VisualInspection/Video.avi
# source ecen_venv/bin/activate