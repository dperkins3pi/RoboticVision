import cv2 as cv
import numpy as np

# Open the video file
cap = cv.VideoCapture('Video.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
    
# Get video properties (like frame width, height, and FPS) and Create VideoWriter object
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)
print("Width:", frame_width, "Height", frame_height)

all_old_points = []
all_new_points = []


METHOD = "knnMatch"
# METHOD = "matchTemplate"
    
if METHOD == "knnMatch":
    out = cv.VideoWriter('processed_video2.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, 
                    (frame_width, frame_height))
    out2 = cv.VideoWriter('processed_video2_nolines.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, 
                    (frame_width, frame_height))
    # Initiate SIFT detector and BFMatcher
    sift = cv.SIFT_create(nfeatures=1000)
    bf = cv.BFMatcher()
    
    frame_number = 0
    skip = 5
    prev_kp = [None]*skip
    prev_des = [None]*skip
    max_distance = 200

    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret: break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        kp1, des1 = prev_kp[frame_number % skip], prev_des[frame_number % skip]
        kp2, des2 = sift.detectAndCompute(frame_gray, None)
        
        if des1 is not None:
            
            # find the keypoints and descriptors with SIFT
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Choose only the best matches
            all_good = []
            for m, n in matches:
                if m.distance < .75*n.distance:
                    all_good.append([m])
                    
            all_good.sort(key=lambda x: x[0].distance)  # Sort matches by distance
            top_n = 500  # Limit the number of matches to top N
            good = all_good[:top_n]  # Keep only the top N matches
            
            # Get the coordinates of the good matches in both images
            points1 = np.float32([kp1[m[0].queryIdx].pt for m in good])
            points2 = np.float32([kp2[m[0].trainIdx].pt for m in good])
            
            all_new_points.append([])
            all_old_points.append([])
            frame2 = frame.copy()
            
            # Draw motion lines
            for new, old in zip(points1, points2):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                
                # Calculate the distance between the points
                dist = np.linalg.norm([a - c, b - d])
                # print(dist)

                # Only draw if the distance is smaller than the threshold
                if dist < max_distance:
                    all_new_points[-1].append([a, b])
                    all_old_points[-1].append([c, d])
                    cv.line(frame, (a, b), (c, d), (0, 255, 0), 2)
                cv.circle(frame, (a, b), 3, (0, 0, 255), -1)
                cv.circle(frame2, (a, b), 3, (0, 0, 255), -1)
                
            # # Display the frame
            # cv.imshow("Feature Matching", frame)
            # cv.waitKey(0)  # Wait indefinitely until a key is pressed
            out.write(frame)
            out2.write(frame2)
            all_new_points[-1] = np.array(all_new_points[-1])
            all_old_points[-1] = np.array(all_old_points[-1])
        
        prev_kp[frame_number % skip], prev_des[frame_number % skip] = kp2, des2
        frame_number += 1
        if frame_number % 20 == 0: print(f"Processed frame {frame_number}")
        # break
else:
    out = cv.VideoWriter('processed_video3.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, 
                    (frame_width, frame_height))
    frame_number = 0
    skip = 5
    template_size = 20
    all_prev_points = [None]*skip
    old_loc = [[None]*400 for _ in range(skip)]
    
    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret: break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        prev_points = all_prev_points[frame_number % skip]
        new_pts = cv.goodFeaturesToTrack(frame_gray, maxCorners=400, qualityLevel=0.01, minDistance=10)
        
        if prev_points is not None:
            for i, point in enumerate(prev_points):
                x, y = point.ravel()  # Get the coordinates of the feature
                
                # Define template size (e.g., 20x20 or smaller)
                x_min = max(int(x) - template_size // 2, 0)
                y_min = max(int(y) - template_size // 2, 0)
                x_max = min(int(x) + template_size // 2, frame_gray.shape[1])
                y_max = min(int(y) + template_size // 2, frame_gray.shape[0])

                # Extract the template (small patch around the point)
                template = frame_gray[y_min:y_max, x_min:x_max]
                
                if template.shape[0] > 0 and template.shape[1] > 0:
                    # Define a small search window around the feature point
                    window_size = 40  # Size of the window for the search area (adjust as needed)
                    x_min_search = max(int(x) - window_size // 2, 0)
                    y_min_search = max(int(y) - window_size // 2, 0)
                    x_max_search = min(int(x) + window_size // 2, frame_gray.shape[1])
                    y_max_search = min(int(y) + window_size // 2, frame_gray.shape[0])
                    
                    # Crop the frame to the search window
                    search_window = frame_gray[y_min_search:y_max_search, x_min_search:x_max_search]

                    if search_window.shape[0] > 0 and search_window.shape[1] > 0:
                        # Perform template matching within the search window
                        result = cv.matchTemplate(search_window, template, cv.TM_CCOEFF_NORMED)

                        # Get the best match location
                        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

                        # Adjust the location relative to the original frame
                        top_left = (max_loc[0] + x_min_search, max_loc[1] + y_min_search)
                        h, w = template.shape[:2]

                        a, b = (top_left[0] + w // 2, top_left[1] + h // 2)
                        
                        # Find the closest previous location in old_loc
                        prev_loc = old_loc[frame_number % skip]
                        distances = np.linalg.norm(np.array(prev_loc) - np.array([a, b]), axis=1)
                        min_index = np.argmin(distances)
                        c, d = prev_loc[min_index]  # Closest point in old_loc
                        
                        cv.circle(frame, (a, b), 3, (0, 0, 255), -1)
                        cv.line(frame, (a, b), (c, d), (0, 255, 0), 2)
                        
                        old_loc[frame_number % skip][i] = (a, b)
            out.write(frame)
        else:
            old_loc[frame_number % skip] = [(0, 0)]*400 # First few frames will be off (and ignored)
        
        all_prev_points[frame_number % skip] = new_pts  # Save the current frame for future use
        # prev_kp[frame_number % skip], prev_des[frame_number % skip] = kp2, des2
        frame_number += 1
        if frame_number % 20 == 0: print(f"Processed frame {frame_number}")
        # break

    
cap.release()
out.release()
cv.destroyAllWindows()

parameters = {'old_points': all_old_points, "new_points": all_new_points}
np.save('matched_points2.npy', parameters)