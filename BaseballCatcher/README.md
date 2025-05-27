# Project Description
I built a system to automatically track and catch a baseball. We caught 17/20 balls—even when they were launched at nearly 60 mph! A video can be found at https://youtu.be/q1axb1CDXnk

⚡ The challenge? We had only a fraction of a second to detect, track, and predict the ball’s landing point.
🔍 How it works:
 ✅ Stereo Vision Calibration: Calibrate a stereo camera system to determine the precise rotation and translation between the two cameras.
 📸 Real-time Ball Detection: Capture images using the stereo system
 🎯 Contour Mapping: Dynamically track the ball using a contout map on the absolute difference between images
 🛠 3D Positioning: Map 2D pixel coordinates into 3D space using stereo calibration parameters.
 📈 Trajectory Prediction: Apply a 2nd-degree polynomial interpolation to estimate the ball’s flight path.
 🤖 Automated Catching: Position the catcher to intercept the ball at the predicted impact point.

![Tracking A Baseball With A Stereo System](BaseballCatcher/TrackingExample2.jpg)

[![Catching A Baseball With A Machine](BaseballCatcher/CatchingABaseball.jpg)](https://youtu.be/q1axb1CDXnk)

# Stereo Calibration
The first step of the project was to calibrate a stereo system of cameras. I gathered three sets of images:
- Chess boards for the left camera
- Chess boards for the right camera
- Chess boards for both cameras
The first two sets were used to find the intrinsic and distortion parameters of each individual camera. The last one was to find the extrinsic parameters (rotation matrix and translation vector) between the two cameras.

After the images were captured, I used OpenCV to calibrate each image (similar to the Calibration project). I also used the function stereoCalibrate() to get the fundamental and essential matrices. Furthermore, I used the stereo calibration parameters to rectify each image, so that the epipolar lines are horizontal. An image showing the absolute differences after rectification is shown below:

![Rectification](StereoCalibration/RectifiedImages.png)

# 3D Trajectory
The next step was to design a system to track the baseball on each camera and then use the coordinates to estimate the 3D trajectory of the moving baseball.

**Tasks 1 and 2:** In the first two tasks, I used and cv.triangulatePoints() and cv.perspectiveTransform() to map the 2D coordinates of the points of the chessboard into a 3-D domain.

**Task 3:** In this task, the goal was to find the pixel coordinates of the baseball in each frame of a video of one being launched towards the camera. I used OpenCV to find the absolute difference between each image. Then, I found the largest contour of the absolute difference that met a specified threshold. Furthermore, to reduce computation time and increase accuracy, I defined a region of interest in each image based on the location of the baseball in the previous frame. An example output for one frame in this task is shown below:

![Baseball Detection](3DTrajectory/BaseballDetection.png)