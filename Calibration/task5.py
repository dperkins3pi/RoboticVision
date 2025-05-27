import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Button, BOTH, Label, Radiobutton  # Graphical User Interface Stuff
from tkinter import font as tkFont
import tkinter as tk
import os

# Button Definitions
ORIGINAL = 0
TAKE_PHOTO = 1
TAKE_AND_STORE_PHOTOS = False   # TODO: Make true if you want to take more photos
CHESS_BOARD_SIZE = (9, 7)
IMAGE_SIZE = (640, 480)
image_folder = "my_chessboard_photos"  # CHANGE THIS BASED ON FILE PATH
display_corners = True   # Change this to true for first task


if TAKE_AND_STORE_PHOTOS:
    camera = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not camera.isOpened(): 
        print("Error: Camera not accessible.")
    else: 
        print("Camera opened successfully.")
    width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    photo_counter = 0  # Keeps track of the photo number

    os.makedirs("my_chessboard_photos", exist_ok=True)

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

        self.root.wm_title(winname)
        positionRight = int(self.root.winfo_screenwidth() / 2 - width / 2)
        positionDown = int(self.root.winfo_screenheight() / 2 - height / 2)
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

        global btnTakePhoto
        btnTakePhoto = Button(text="Take Photo", command=self.take_photo)
        btnTakePhoto['font'] = helv18
        btnTakePhoto.pack(side='left', pady=2)

        # radio buttons
        global mode
        mode = tk.IntVar()
        mode.set(ORIGINAL)

        # threading
        self.stopevent = threading.Event()
        self.thread = threading.Thread(target=self.capture, args=())
        self.thread.start()

    def capture(self):
        while not self.stopevent.is_set():
            if not self.stopflag:
                ret0, frame = camera.read()

                # Update image
                image = cvMat2tkImg(frame)
                self.panel.configure(image=image)
                self.panel.image = image

    def take_photo(self):  # This function will trigger photo capture
        global photo_counter
        mode.set(TAKE_PHOTO)  # Set mode to TAKE_PHOTO

        ret0, frame = camera.read()  # Capture frame
        photo_counter += 1
        filename = os.path.join("my_chessboard_photos", f'photo_{photo_counter}.jpg')
        cv.imwrite(filename, frame)  # Save the frame as a photo
        print(f"Photo saved as {filename}")
        self.display_saved_photo(filename)
        
        mode.set(ORIGINAL)  # After taking the photo, set the mode back to ORIGINAL

    def display_saved_photo(self, filename):
        img = Image.open(filename)
        img = img.resize((width // 2, height // 2))  # Resize for display
        img = ImageTk.PhotoImage(img)
        self.panel.configure(image=img)
        self.panel.image = img

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

if TAKE_AND_STORE_PHOTOS:
    print("Open app")
    app = App()
    print("Running app")
    app.run()
    print("Releasing the Camera")
    # release the camera
    camera.release()
    

# Load in the data
files = os.listdir(image_folder)
color_images = []  # Lists of (480, 640, 3) color images
gray_images = []  # Lists of (480, 640) gray images
for file in files:
    if "jpg" in file:  # Only load in certain images
        image = cv.imread(os.path.join(image_folder, file))
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Get gray scale image
        color_images.append(image)
        gray_images.append(gray_image)
        
# Get object points array for one arbitrary image
one_object_point = np.zeros((CHESS_BOARD_SIZE[0] * CHESS_BOARD_SIZE[1], 3), np.float32)
one_object_point[:, :2] = np.mgrid[0:CHESS_BOARD_SIZE[0], 0:CHESS_BOARD_SIZE[1]].T.reshape(-1, 2)

# Initialize object and image points
object_points = []
image_points = []

# Find the corners
for i, image in enumerate(gray_images):
    # Find the chess board corners (Corners outputs a list of all the detected corners)
    ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_SIZE, None)  # 10x7 internal corners
    
    # Restart if the corners weren't found
    if not ret:
        print("Corners not found in", files[i])
        continue
    
    # Refine the corner locations using cornerSubPix
    corners_refined = cv.cornerSubPix(image, corners, (11, 11), (-1, -1), 
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    # Output is num_corners x 1 x 2 ((x,y) coordinates at each corner)
    
    # Store object an dimage points
    object_points.append(one_object_point)  # 3D real-world points
    image_points.append(corners_refined.reshape(-1, 2))  # 2D refined image points

    # Display first image if desired
    if i == 1 and display_corners:
        image_with_corners = cv.drawChessboardCorners(color_images[i], CHESS_BOARD_SIZE, corners_refined, ret)
        # Display the image with the corners drawn
        cv.imshow('Chessboard Corners', image_with_corners)
        cv.waitKey(0)

# Calculate the intrinsic and distortion parameters. 
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, IMAGE_SIZE, None, None)
# ret: The overall calibration error, a scalar.
# mtx: The camera matrix K (3x3), containing the intrinsic parameters (focal length, principal point).
# dist: The distortion coefficients (5 parameters: k1, k2, p1, p2, k3).
# rvecs: Rotation vectors for each image, representing the orientation of the camera relative to the calibration pattern.
    # The output here is the angles, not the rotation matrix
# tvecs: Translation vectors for each image, representing the position of the camera relative to the calibration pattern.

# Get the focal length in pixels
fx, fy = mtx[0, 0], mtx[1,1]
# Convert the unit from pixels to mm
fx, fy = fx/135, fy/135    # 135 = 648/4.8

# Print out intrinsic and distortion parameters
print("Focal Length:", fx, "or", fy)
print("Intrinsic Parameters:\n", mtx)
print("Distortion Parameters:", dist)

# Save parameters into a file
parameters = {'ret': ret, 
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs}
np.save('task5_parameters.npy', parameters)
