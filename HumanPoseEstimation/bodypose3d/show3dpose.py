import cv2 as cv
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
plt.style.use('seaborn-v0_8-whitegrid')

# Make sure these keypoints match that used in bodypose2d.py
pose_keypoints = [i for i in range(33)]

def read_keypoints(filename):
    """Read in the file into an np.arrau

    Args:
        filename (Str): Location of the file

    Returns:
        kpts (np.array) num_framesx33x3: x,y,z coordinates of each joint at every frame
    """
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (len(pose_keypoints), -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts

def render_figure_to_cvimg(fig):
    """Render a Matplotlib figure to an OpenCV image (BGR)"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv.imdecode(img_array, cv.IMREAD_COLOR)
    return img

def visualize_3d(p3ds, title, save_video=False, save_path='3d_output.mp4'):
    """Visalize the joints in 3-dimensions

    Args:
        p3ds (np.array): num_framesx33x3: x,y,z coordinates of each joint at every frame
        title (str): Title to give the plot
        save_video (bool): Whether or not to save the video as a file
            - Currently the code for saving the video is quite slow and not optimized at all
            - It could be greatly sped up using matplotlib.animation.FuncAnimation
        save_path (str): Location to save the 3D video
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    writer = None
    
    # Define body part groups by MediaPipe connection indices
    body_colors = {
        "face": 'orange',
        "left_arm": 'blue',
        "right_arm": 'blue',
        "left_leg": 'green',
        "right_leg": 'green',
        "torso": 'gray',
        "hands": 'magenta',
        "feet": 'brown'
    }

    # Group connections manually by region
    face_connections = set(range(0, 11)) | {7, 8}
    left_arm = {11, 13, 15, 17, 19, 21}
    right_arm = {12, 14, 16, 18, 20, 22}
    left_leg = {23, 25, 27, 29, 31}
    right_leg = {24, 26, 28, 30, 32}
    torso = {11, 12, 23, 24}
    hands = {15, 17, 19, 21, 16, 18, 20, 22}
    feet = {27, 29, 31, 28, 30, 32}
    
    # Define a function to get the colors of each body part
    def get_color(idx1, idx2=None):
        check_set = {idx1} if idx2 is None else {idx1, idx2}
        if check_set & face_connections: return body_colors["face"]
        elif check_set & left_arm: return body_colors["left_arm"]
        elif check_set & right_arm: return body_colors["right_arm"]
        elif check_set & left_leg: return body_colors["left_leg"]
        elif check_set & right_leg: return body_colors["right_leg"]
        elif check_set & hands: return body_colors["hands"]
        elif check_set & feet: return body_colors["feet"]
        elif check_set & torso: return body_colors["torso"]
        else: return 'black'

    for framenum, kpts3d in enumerate(p3ds):
        
        if framenum % 2 == 0:  # Only store every other frame in the video (for higher speed)
            continue

        # Draw connections
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx >= kpts3d.shape[0] or end_idx >= kpts3d.shape[0]:
                continue
            xs = [kpts3d[start_idx, 0], kpts3d[end_idx, 0]]
            ys = [-kpts3d[start_idx, 2], -kpts3d[end_idx, 2]]
            zs = [-kpts3d[start_idx, 1], -kpts3d[end_idx, 1]]
            color = get_color(start_idx, end_idx)
            ax.plot(xs, ys, zs, c=color, linewidth=2)

        # Draw keypoints
        for i in range(kpts3d.shape[0]):
            ax.scatter(xs=kpts3d[i:i+1, 0], ys=-kpts3d[i:i+1, 2], zs=-kpts3d[i:i+1, 1], color=get_color(i), s=20)

        # Set up the plot
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Set limits to the plot
            # TODO: These numbers and hard coded in and need to be changed for new videos
        ax.set_xlim3d(-50, 30)
        ax.set_ylim3d(-290, -190)
        ax.set_zlim3d(-60, 40)

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_title(title)
        
        # Save to video
        if save_video:
            fig.canvas.draw()
            # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            cv_frame = render_figure_to_cvimg(fig)

            if writer is None:
                height, width = cv_frame.shape[:2]
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                writer = cv.VideoWriter(save_path, fourcc, 10.0, (width, height))

            writer.write(cv_frame)

        if not save_video: plt.pause(0.001)
        ax.cla()
        
    if writer is not None:
        writer.release()
        print(f"[INFO] Video saved to: {save_path}")

def gaussian_smooth_points(p3ds, sigma=2.5):
    """
    Apply Gaussian smoothing to the 3D points over time.
    
    Parameters:
    - p3ds: (n_frames, n_points, 3) matrix of 3D keypoints over time.
    - sigma: Standard deviation of the Gaussian kernel (higher values smooth more).
    
    Returns:
    - smoothed_p3ds: Smoothed 3D points.
    """
    smoothed_p3ds = np.copy(p3ds)
    for i in range(p3ds.shape[1]):  # Loop over keypoints
        for j in range(3):  # Loop over x, y, z
            smoothed_p3ds[:, i, j] = gaussian_filter1d(p3ds[:, i, j], sigma=sigma)
    return smoothed_p3ds

if __name__ == '__main__':

    method1 = True   # True if the coordinates are from the DLT method
    smooth = False    # True if you want to smooth the coordinates (to improve visualization)
    action = "squat"  # Replace with 'chilling', 'chilling2', 'huluhoop', 'jogging', 'jumpingjacks', 'ninjamoves', or 'squat'
    save_video = False   # Whther or not to save the video (greatly affects the speed of the code)
    
    # Get the file path (change this code if the file is in a different location) and read in the coordinates
    if method1: file_path = "data/" + action + "/kpts_3d.dat"
    else: file_path = "data/" + action + "/kpts_3d2.dat"
    p3ds = read_keypoints(file_path)
    
    # Makt the video
    print("Making the video at", file_path)
    if smooth: 
        path = "data/" + action + "/3D_smoothed.mp4"
        visualize_3d(gaussian_smooth_points(p3ds), title=action, save_video=save_video, save_path=path)
    else: 
        path = "data/" + action + "/3D.mp4"
        visualize_3d(p3ds, title=action, save_video=save_video, save_path=path)
        
        
    # If you want to run the model on all actions and both smoothing methods, replace the code above with the code below
    
    # for smooth in [True, False]:
    #     for action in ["squat", "huluhoop", "jogging", "jumpingjacks", "ninjamoves", "chilling", "chilling2"]:
    #         if method1: file_path = "data/" + action + "/all_kpts_3d.dat"
    #         else: file_path = "data/" + action + "/all_kpts_3d2.dat"
    #         p3ds = read_keypoints(file_path)
    #         print("Making the video at", file_path)
    #         if smooth: 
    #             path = "data/" + action + "/3D_smoothed.mp4"
    #             visualize_3d(gaussian_smooth_points(p3ds), title=action, save_video=save_video, save_path=path)
    #         else: 
    #             path = "data/" + action + "/3D.mp4"
    #             visualize_3d(p3ds, title=action, save_video=save_video, save_path=path)