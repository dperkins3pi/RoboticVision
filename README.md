# RoboticVision
Various computer vision projects from BYU's grad course ECEN 631: Robotic Vision. Projects use openCV for topics such as stereo calibration, catching a baseball, visual inspection, and more.

# Visual Inspection
In this project, the goal was to create software that automatically classifies objects on a conveyer belt into three categories (Good, Bad, and Ugly) in real time. Specifically, we worked with Babybel cheese where "good" meant unopened, "bad" meant opened, and "ugly" meant partially eaten. Using old school computer vision techniques, contour maps of each frame were taken to find the location of the cheese. Features such as the texture, shape, and color were extracted and passed into an SVM to classify the object.

For this project, we manually gathered our own data by recording videos of the cheese passing by on a conveyer belt at various speeds. We hand-crafted our own features and then used them to train the SVM. 

We were able to achieve 92% accuracy on the training set and had good results on real-time tests. However, the accuracy ended up being highly dependent on the lighting conditions, emphasizing the need for more data or a controlled environment.