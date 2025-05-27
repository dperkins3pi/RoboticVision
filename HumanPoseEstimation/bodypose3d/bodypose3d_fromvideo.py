import cv2 as cv
import mediapipe as mp
import numpy as np
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Calibrataion parameters for the stereo system (in ECEN 631 lab)
mtxL = np.array([[1.68788706e+03, 0.00000000e+00, 3.26074785e+02], 
                [0.00000000e+00, 1.69033873e+03, 2.29100136e+02], 
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distL = np.array([[-4.88594653e-01, -1.24679131e+00, 3.89316951e-03,  7.85455406e-04, 3.05969397e+01]])
mtxR = np.array([[1.69280103e+03, 0.00000000e+00, 3.22865638e+02],
                [0.00000000e+00, 1.69707855e+03, 2.01990927e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distR = np.array([[-5.33564467e-01, 5.23309648e+00, 5.25438881e-03, 3.13336667e-03, -1.15454612e+02]])
T = np.array([[-20.38998479],[-0.02986332], [0.61417699]])
R0 = np.array([[ 0.9993875, 0.00516153, -0.03461194],
            [-0.00487561, 0.99995333, 0.00834004],
            [0.03465337, -0.00816618, 0.99936603]])
R1 = np.array([[0.99954558, 0.00146394, -0.03010782],
            [-0.00171244, 0.99996467, -0.00822953],
            [0.0300947, 0.00827735, 0.99951278]])
P0 = np.array([[1.69377887e+03, 0.00000000e+00, 3.82358589e+02, 0.00000000e+00],
    [0.00000000e+00, 1.69377887e+03, 2.14102676e+02, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])
P1 = np.array([[1.69377887e+03, 0.00000000e+00, 3.82358589e+02, -3.45518263e+04],
    [ 0.00000000e+00, 1.69377887e+03, 2.14102676e+02, 0.00000000e+00],
    [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])


# The frame shape may need to be editted, based on the input from the cameras
frame_shape = [480, 640]
pose_keypoints = [i for i in range(33)]  # Change if you don't want to see all keypoints
    # Information about the specific keypoints can be found at https://roboflow.com/model/mediapipe


def write_keypoints_to_disk(filename, kpts):
    """Write the coordinates to a file for storage

    Args:
        filename (str): Filename
        kpts (np.array): Points to store
    """
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
            else:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

        fout.write('\n')
    fout.close()
    
    
def DLT(point1, point2):
    """Get 3-D coordinates form 2D ones in each frame using DLT method

    Args:
        point1 (np.array): Pixel coordinates of the joint (from MediaPipe)
        point2 (np.array): Pixel coordinates of the joint (from MediaPipe)
        
    Returns:
        p3d (np.array): 3-D coordinates of the joint
    """
    A = [point1[1]*P0[2,:] - P0[1,:],
        P0[0,:] - point1[0]*P0[2,:],
        point2[1]*P1[2,:] - P1[1,:],
        P1[0,:] - point2[0]*P1[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
    
    return Vh[3,0:3]/Vh[3,3]

def get_3DCoordinates_OpenCV(uv1, uv2):
    """Get 3-D coordinates form 2D ones in each frame using our OpenCV method

    Args:
        P1 (np.array): Projection Matrix 1
        P2 (np.array): Projection Matrix 2
        point1 (np.array): Pixel coordinates of the joint (from MediaPipe)
        point2 (np.array): Pixel coordinates of the joint (from MediaPipe)
        
    Returns:
        p3d (np.array): 3-D coordinates of the joint
    """
    # Reshape points for undistortion
    uv1_reshaped = np.array(uv1, dtype=np.float32).reshape(1, 1, 2)  # Assuming uv1 is a 2D point
    uv2_reshaped = np.array(uv2, dtype=np.float32).reshape(1, 1, 2)  # Assuming uv2 is a 2D point

    # Undistort the points
    undistorted_uv1 = cv.undistortPoints(uv1_reshaped, mtxL, distL, R=R0, P=P0)
    undistorted_uv2 = cv.undistortPoints(uv2_reshaped, mtxR, distR, R=R1, P=P1)

    # Triangulation (homogeneous 3D coordinates)
    homogeneous_point_3d = cv.triangulatePoints(P0, P1, undistorted_uv1, undistorted_uv2)
    
    # Convert homogeneous coordinates to 3D coordinates (divide by the last element)
    _p3d2 = homogeneous_point_3d[:3] / homogeneous_point_3d[3]
    return _p3d2


def run_mp(input_stream1, input_stream2, show_lines=True, save_video=False):
    """Run google's mediapipe to get 2-D pixel coordinates and 3-D coordinates of the joints

    Args:
        input_stream1 (str): Location to the video in the first camera
        input_stream2 (_type_): Location of the video on the second camera
        show_lines (bool): Whether or not to show lines in the plot
        save_video (bool): Whether or not to save the video

    Returns:
        kpts_cam0 (np.array) num_framesx33x2: x,y coordinates of each joint at every frame in first camera
        kpts_cam1 (np.array) num_framesx33x2: x,y coordinates of each joint at every frame in second camera
        kpts_3d (np.array) num_framesx33x3: x,y,z coordinates of each joint at every frame using DLT method
        kpts_3d2 (np.array) num_framesx33x3: x,y,z coordinates of each joint at every frame using OpenCV method
    """
    # Put camera id as command line arguments
    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])
    
    out0 = None
    out1 = None
        
    # Input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    # Set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    # Create body keypoints detector objects.
    pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Containers for detected keypoints for each camera. These are filled at each frame.
    # This will run you into memory issue if you run the program without stop
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []
    kpts_3d2 = []
    
    while True:
        # R frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1: break
        
        # Initialize video writers after reading the first valid frame
        if save_video and out0 is None and ret0 and ret1:
            height, width = frame0.shape[:2]            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out0 = cv.VideoWriter('left_camera.mp4', fourcc, 20.0, (frame_shape[0], frame_shape[0]))
            out1 = cv.VideoWriter('right_camera.mp4', fourcc, 20.0, (frame_shape[0], frame_shape[0]))

        # Crop to 720x720.
        # Note: camera calibration parameters are set to this resolution. If you change this, make sure to also change camera intrinsic parameters
        if frame0.shape[1] != 720:
            frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # Convert Image to RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # Pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = pose0.process(frame0)
        results1 = pose1.process(frame1)

        # Reverse changes
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        # Check for keypoints detection
        frame0_keypoints = []
        if results0.pose_landmarks:
            for i, landmark in enumerate(results0.pose_landmarks.landmark):
                if i not in pose_keypoints: continue # Only save keypoints that are indicated in pose_keypoints
                pxl_x = landmark.x * frame0.shape[1]
                pxl_y = landmark.y * frame0.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame0, (pxl_x, pxl_y), 3, (0,0,255), -1) # Add keypoint detection points into figure
                kpts = [pxl_x, pxl_y]
                frame0_keypoints.append(kpts)
        else:
            # If no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_keypoints = [[-1, -1]]*len(pose_keypoints)

        # This will keep keypoints of this frame in memory
        kpts_cam0.append(frame0_keypoints)

        frame1_keypoints = []
        if results1.pose_landmarks:
            for i, landmark in enumerate(results1.pose_landmarks.landmark):
                if i not in pose_keypoints: continue
                pxl_x = landmark.x * frame1.shape[1]
                pxl_y = landmark.y * frame1.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame1,(pxl_x, pxl_y), 3, (0,0,255), -1)
                kpts = [pxl_x, pxl_y]
                frame1_keypoints.append(kpts)
        else:
            # If no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame1_keypoints = [[-1, -1]]*len(pose_keypoints)

        # Update keypoints container
        kpts_cam1.append(frame1_keypoints)

        # Calculate 3d position
        frame_p3ds = []
        frame_p3ds2 = []
        
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):

            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                # Calculate 3d position of keypoint (using both methods)
                _p3d = DLT(uv1, uv2)
                _p3d2 = get_3DCoordinates_OpenCV(uv1, uv2)
                
            frame_p3ds.append(_p3d)
            frame_p3ds2.append(_p3d2)

        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        frame_p3ds = np.array(frame_p3ds).reshape((len(pose_keypoints), 3))
        kpts_3d.append(frame_p3ds)
        frame_p3ds2 = np.array(frame_p3ds2).reshape((len(pose_keypoints), 3))
        kpts_3d2.append(frame_p3ds2)

        if show_lines:
            mp_drawing.draw_landmarks(frame0, results0.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv.imshow('cam1', frame1)
        cv.imshow('cam0', frame0)
        
        if save_video:
            out0.write(frame0)
            out1.write(frame1)

        k = cv.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    if save_video: 
        out0.release()
        out1.release()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d), np.array(kpts_3d2)


if __name__ == '__main__':

    write_output = False   # Whether or not to write the coordinates into kpts_cam file
    save_video = False  # Whether or not to save the video
    
    # Replace input_stream1 and input_stream2 with the location of the video
    action = "jumpingjacks"   # Replace with 'chilling', 'chilling2', 'huluhoop', 'jogging', 'jumpingjacks', 'ninjamoves', or 'squat'
    input_stream1 = 'data/' + action + '/left_original.avi'
    input_stream2 = 'data/' + action + '/right_original.avi'

    # Run MediaPipe
    kpts_cam0, kpts_cam1, kpts_3d, kpts_3d2 = run_mp(input_stream1, input_stream2, show_lines=True, save_video=save_video)

    # Create keypoints file in current working folder
    if write_output:
        write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
        write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
        write_keypoints_to_disk('kpts_3d.dat', kpts_3d)
        write_keypoints_to_disk('kpts_3d2.dat', kpts_3d2)
