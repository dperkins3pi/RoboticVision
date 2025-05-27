import cv2 as cv
import mediapipe as mp
import numpy as np
import sys

# Initialize MediaPipe drawing and pose modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Define frame resolution and keypoints to extract (all 33 pose landmarks)
frame_shape = [480, 640]
pose_keypoints = [i for i in range(33)]

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

def run_mp(input_stream1, show_lines=True, save_video=False):
    """
    Run 2D pose estimation on a single webcam feed.
    
    Args:
        input_stream1 (int or str): Camera index (e.g., 0) or video file path.
        show_lines (bool): Whether to draw pose connections on the frame.
        save_video (bool): Whether to save the output video.
    
    Returns:
        kpts_cam (np.ndarray): Array of 2D keypoints for each frame.
    """
    if len(sys.argv) == 2:
        input_stream1 = int(sys.argv[1])
    
    # Initialize video writer as None
    out0 = None
    
    # Set up the single camera capture
    cap0 = cv.VideoCapture(input_stream1, cv.CAP_DSHOW)   # Try removing cv.CAP_DSHOW if it isn't working (the flag makes it work for Windows)
    cap0.set(3, frame_shape[1])  # Width: 640
    cap0.set(4, frame_shape[0])  # Height: 480
    
    # Initialize MediaPipe Pose
    pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # List to store 2D keypoints for all frames
    kpts_cam = []
    
    while True:
        # Read frame from the webcam
        ret0, frame0 = cap0.read()
        if not ret0:
            break
        
        # Initialize video writer on the first frame if saving is enabled
        if save_video and out0 is None:
            height, width = frame0.shape[:2]
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out0 = cv.VideoWriter('camera.mp4', fourcc, 20.0, (width, height))
        
        # Process the frame for pose estimation
        frame0_rgb = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame0_rgb.flags.writeable = False
        results0 = pose0.process(frame0_rgb)
        frame0_rgb.flags.writeable = True
        frame0 = cv.cvtColor(frame0_rgb, cv.COLOR_RGB2BGR)
        
        # Extract 2D keypoints from pose landmarks
        frame0_keypoints = []
        if results0.pose_landmarks:
            for i, landmark in enumerate(results0.pose_landmarks.landmark):
                if i not in pose_keypoints:
                    continue
                pxl_x = int(round(landmark.x * frame0.shape[1]))  # Scale to frame width
                pxl_y = int(round(landmark.y * frame0.shape[0]))  # Scale to frame height
                cv.circle(frame0, (pxl_x, pxl_y), 3, (0, 0, 255), -1)  # Draw keypoint
                kpts = [pxl_x, pxl_y]
                frame0_keypoints.append(kpts)
        else:
            # If no landmarks detected, fill with invalid coordinates
            frame0_keypoints = [[-1, -1]] * len(pose_keypoints)
        
        # Store keypoints for the current frame
        kpts_cam.append(frame0_keypoints)
        
        # Draw pose connections if enabled
        if show_lines:
            mp_drawing.draw_landmarks(
                frame0,
                results0.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Display the frame
        cv.imshow('cam0', frame0)
        
        # Write frame to video file if saving is enabled
        if save_video:
            out0.write(frame0)
        
        # Exit on ESC key press
        k = cv.waitKey(1)
        if k & 0xFF == 27:
            break
    
    # Cleanup
    cv.destroyAllWindows()
    cap0.release()
    if save_video and out0 is not None:
        out0.release()
    
    return np.array(kpts_cam)

if __name__ == '__main__':
    
    # Configuration
    input_stream1 = 0    # Default to webcam index 0, allow override via command-line argument
    write_output = True  # Set to True to save 2D keypoints to disk
    save_video = True   # Set to True to save the video
    
    # Run the pose estimation
    kpts_cam = run_mp(input_stream1, show_lines=True, save_video=save_video)
    
    # Save 2D keypoints to disk
    if write_output:
        write_keypoints_to_disk('kpts_cam.dat', kpts_cam)
        print("2D keypoints saved to 'kpts_cam.dat'")