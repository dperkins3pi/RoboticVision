# RoboticVision
Various computer vision projects from BYU's grad course ECEN 631: Robotic Vision. Projects use openCV for topics such as stereo calibration, catching a baseball, visual inspection, and more.

# Visual Inspection
In this project, the goal was to create software that automatically classifies objects on a conveyer belt into three categories (Good, Bad, and Ugly) in real time. Specifically, we worked with Babybel cheese where "good" meant unopened, "bad" meant opened, and "ugly" meant partially eaten. We...

- Manually gathered our own data by recording videos of the cheese passing by on a conveyer belt at various speeds
- Hand-crafted our own features using old school computer vision methods on OpenCV
- Trained an SVM on the gathered data
- Achieved 92% accuracy on the training set and had good results on real-time tests.

<p align="center">
  <img src="VisualInspection/SampleImage.png" alt="Sample Output" width="400"/>
</p>

# MotionField
In this project, the goal was to understand the motion of a moving camera in a feature-rich scene. I used OpenCV to detect useful features in each frame and then matched them from frame to frame (with a 5-frame buffer) using various OpenCV methods. I then visualized the movement by displaying the features and their trajectories. I also calculated the rotation matrices and translation vectors for each movement. An example image is shown below.

<p align="center">
  <img src="MotionField/Rotation.png" alt="Rotation" width="400"/>
</p>

# TimeToImpact
In this project, a series of images were captured from a camera moving toward a spray can. The objective was to estimate the **time to impact**—the moment the camera would collide with the spray can.

To achieve this, I applied classical computer vision techniques:
- Extracted key features from a reference image of the spray can
- Matched these features across frames to locate the spray can in each image
- Calculated a bounding box around the matched region in each frame
- Analyzed the change in bounding box size over time to estimate the camera's trajectory and predict the time to impact

This approach provided a simple yet effective way to model motion and depth perception using only 2D visual input.

<p align="center">
  <img src="TimeToImpact/Features_Found/frame_17_all_features.jpg" alt="Sample Image" width="400"/>
</p>

# Image Classification
For this task, the goal was to classify various fish species and oyster shells. Since I am already familiar with YOLO, I chose to use a ResNet50. Sample images and results for the fish classification are below:

<p align="center">
  <img src="ImageClassification/Fish.png" alt="Sample Image" width="400"/>
</p>

<p align="center">
  <img src="ImageClassification/conf_mat_fish.png" alt="Confusion Matrix" width="400"/>
</p>

# openCV_Intro
This project was an intro to OpenCV. I learned how to use various openCV functions in real time, such as edge detection, line detection, corner detection, absolute difference, and contour maps. Also, I used these functions to detect images of a launched baseball.

<p align="center">
  <img src="openCV_Intro/EdgeDetection.png" alt="Edge Detection" width="400"/>
</p>
