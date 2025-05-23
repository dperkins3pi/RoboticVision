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