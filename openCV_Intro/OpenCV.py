# Feature Detection (Task 1 and 2): https://youtu.be/lBLe0adD2Jk
# Baseball (Task 3): https://youtu.be/_503ZX4dXfQ

#######################################Feature Detection#######################################


import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Button, BOTH, Label, Scale, Radiobutton  # Graphical User Inetrface Stuff
from tkinter import font as tkFont
import tkinter as tk

print("OpenCV version", cv.__version__)
camera = cv.VideoCapture(0, cv.CAP_DSHOW)
if not camera.isOpened(): print("Error: Camera not accessible.")
else: print("Camera opened successfully.")
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
videoout = cv.VideoWriter('./Video.avi', cv.VideoWriter_fourcc(*'XVID'), 25, (width, height))  # Video format

# Button Definitions
ORIGINAL = 0
BINARY = 1
EDGE = 2
LINE = 3
ABSDIFF = 4
RGB = 5
HSV = 6
CORNER = 7
CONTOUR = 8


def cvMat2tkImg(arr):  # Convert OpenCV image Mat to image for display
    rgb = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(img)


class App(Frame):
    def __init__(self, winname='OpenCV'):  # GUI Design

        self.root = Tk()
        self.stopflag = True
        self.buffer = np.zeros((height, width, 3), dtype=np.uint8)

        global helv18
        helv18 = tkFont.Font(family='Helvetica', size=18, weight='bold')
        # print("Width",windowWidth,"Height",windowHeight)
        self.root.wm_title(winname)
        positionRight = int(self.root.winfo_screenwidth() / 2 - width / 2)
        positionDown = int(self.root.winfo_screenheight() / 2 - height / 2)
        # Positions the window in the center of the page.
        self.root.geometry("+{}+{}".format(positionRight, positionDown))
        self.root.wm_protocol("WM_DELETE_WINDOW", self.exitApp)
        Frame.__init__(self, self.root)
        self.pack(fill=BOTH, expand=1)
        # capture and display the first frame
        ret0, frame = camera.read()
        image = cvMat2tkImg(frame)
        self.panel = Label(image=image)
        self.panel.image = image
        self.panel.pack(side="top")
        # buttons
        global btnQuit
        btnQuit = Button(text="Quit", command=self.quit)
        btnQuit['font'] = helv18
        btnQuit.pack(side='right', pady=2)
        global btnStart
        btnStart = Button(text="Start", command=self.startstop)
        btnStart['font'] = helv18
        btnStart.pack(side='right', pady=2)
        # sliders
        global Slider1, Slider2
        Slider2 = Scale(self.root, from_=0, to=255, length=255, orient='horizontal')
        Slider2.pack(side='right')
        Slider2.set(192)
        Slider1 = Scale(self.root, from_=0, to=255, length=255, orient='horizontal')
        Slider1.pack(side='right')
        Slider1.set(64)
        # radio buttons
        global mode
        mode = tk.IntVar()
        mode.set(ORIGINAL)
        Radiobutton(self.root, text="Original", variable=mode, value=ORIGINAL).pack(side='left', pady=4)
        Radiobutton(self.root, text="Binary", variable=mode, value=BINARY).pack(side='left', pady=4)
        Radiobutton(self.root, text="Edge", variable=mode, value=EDGE).pack(side='left', pady=4)
        Radiobutton(self.root, text="Line", variable=mode, value=LINE).pack(side='left', pady=4)
        Radiobutton(self.root, text="Abs Diff", variable=mode, value=ABSDIFF).pack(side='left', pady=4)
        Radiobutton(self.root, text="RGB", variable=mode, value=RGB).pack(side='left', pady=4)
        Radiobutton(self.root, text="HSV", variable=mode, value=HSV).pack(side='left', pady=4)
        Radiobutton(self.root, text="Corner", variable=mode, value=CORNER).pack(side='left', pady=4)
        Radiobutton(self.root, text="Contour", variable=mode, value=CONTOUR).pack(side='left', pady=4)
        # threading
        self.stopevent = threading.Event()
        self.thread = threading.Thread(target=self.capture, args=())
        self.thread.start()

    def capture(self):
        current_frame = None   # Image buffer
        while not self.stopevent.is_set():
            if not self.stopflag:
                prev_frame = current_frame
                ret0, frame = camera.read()
                current_frame = frame.copy()
                
                if mode.get() == BINARY:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    
                    # My Code Here
                    gray_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale
                    frame = cv.inRange(gray_frame, lThreshold, hThreshold)   # Apply range-based binarization
                    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)  # Convert to 3 layers
                        # dst =  0 if src < low threshold or src > high threshold 
                        # dst =  255 otherwise 

                elif mode.get() == EDGE:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    
                    # My code here
                    frame = cv.Canny(current_frame, lThreshold, hThreshold, True)
                    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)  # Convert to 3 layers
                
                # My code here    
                elif mode.get() == CORNER:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    
                    blockSize = 2 + round(lThreshold * (10 / 255))   # Move it to range 2-12
                    ksize = round(lThreshold * (10 / 255))  # Move it to range 0-10
                    if ksize % 2 == 0: ksize += 3  # Must be an odd number 3 or higher
                    k = 0.001 + hThreshold * (.3 / 255)  # Also alter k
                    
                    # Convert to grayscale and float32
                    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    gray_frame = np.float32(gray_frame)

                    # Apply Harris corner detection and mark th
                    corners = cv.cornerHarris(gray_frame, blockSize, ksize, k)

                    # Dilate the corners to mark them clearly
                    corners = cv.dilate(corners, None)

                    # Mark corners in red on the original frame based on the thresholds
                    frame[corners > 0.01 * corners.max()] = [0, 0, 255]  # Red color for corners

                    
                elif mode.get() == LINE:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    
                    # My code here
                    
                    # Convert to grayscale
                    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                    # Apply Canny edge detection
                    edges = cv.Canny(gray_frame, threshold1=lThreshold, threshold2=hThreshold)

                    # Detect lines using Hough Transform
                    lines = cv.HoughLines(edges, 1, np.pi/180, hThreshold)
                    
                    if lines is not None:  # Check if any lines are detected 
                        for line in lines:   # Display the lines
                            rho, theta = line[0]
                            a, b = np.cos(theta), np.sin(theta)
                            x0, y0 = a*rho, b*rho
                            x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
                            x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))
                            cv.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

                                    
                elif mode.get() == ABSDIFF:
                    
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    
                    # Add your code here
                    if prev_frame is not None: # Perform absolute differencing
                        # Make it gray scale
                        if len(prev_frame.shape) == 3: gray_image_prev = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
                        else: gray_image_prev = prev_frame
                        if len(current_frame.shape) == 3: gray_image = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
                        else: gray_image = current_frame
                        # Find the difference
                        diff = cv.absdiff(gray_image_prev, gray_image)
                        _, frame = cv.threshold(diff, lThreshold, hThreshold, cv.THRESH_BINARY)
                    else: # If no previous frame exists, just copy the current frame (initial case)
                        frame = current_frame.copy()
                    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)  # Convert to 3 layers


                elif mode.get() == CONTOUR:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    
                    # Convert frame to grayscale and apply thresholding
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    _, thresh = cv.threshold(gray, lThreshold, hThreshold, cv.THRESH_BINARY)

                    # Find contours
                    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

                    # Iterate through contours and draw bounding boxes for large objects
                    for contour in contours:
                        area = cv.contourArea(contour)
                        if area > 1000:  # Threshold for large objects (adjust as needed)
                            x, y, w, h = cv.boundingRect(contour)   # Get the bounding box for the contour

                            # Draw the bounding box on the frame
                            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if 1000 < area < 10000:  # Example: Filter small and overly large areas
                            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
                    
                elif mode.get() == RGB:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                elif mode.get() == HSV:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here

                image = cvMat2tkImg(frame)
                self.panel.configure(image=image)
                self.panel.image = image
                prev_frame = frame.copy()   # Store previous frame
                videoout.write(frame)


    def startstop(self):  # toggle flag to start and stop
        if btnStart.config('text')[-1] == 'Start':
            btnStart.config(text='Stop')
        else:
            btnStart.config(text='Start')
        self.stopflag = not self.stopflag

    def run(self):  # run main loop
        self.root.mainloop()

    def quit(self):  # exit loop
        self.stopflag = True
        t = threading.Timer(1.0, self.stop)  # start a timer (non-blocking) to give main thread time to stop
        t.start()

    def exitApp(self):  # exit loop
        self.stopflag = True
        t = threading.Timer(1.0, self.stop)  # start a timer (non-blocking) to give main thread time to stop
        t.start()

    def stop(self):
        self.stopevent.set()
        self.root.quit()


print("Open app")
app = App()
print("Running app")
app.run()
print("Releasing the Camera")
# release the camera
camera.release()


#######################################Baseball#######################################


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