import cv2 as cv
import numpy as np

# Load in images
reference_color = cv.imread('img2_cropped.jpg')
reference  = cv.imread('img2_cropped.jpg', cv.IMREAD_GRAYSCALE)
target_color = cv.imread('NachoLibre.jpg')
target = cv.imread('NachoLibre.jpg', cv.IMREAD_GRAYSCALE)

# Reshape img3 to match the shape of img2
img3_color = cv.resize(target_color, (reference_color.shape[1], reference_color.shape[0]))
img3 = cv.resize(target, (reference.shape[1], reference.shape[0]))

# Open the video file
cap = cv.VideoCapture('video.MOV')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties (like frame width, height, and FPS) and Create VideoWriter object
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)
out = cv.VideoWriter('processed_video.MOV', cv.VideoWriter_fourcc(*'mp4v'), fps, 
                    (frame_width, frame_height))

# Initiate SIFT detector
sift = cv.SIFT_create(nfeatures=2500)
kp1, des1 = sift.detectAndCompute(reference, None)
h, w = reference.shape
start_frame = int(fps * 10)
previous_H = None
previous_image = None
frame_count = 0
failed_count = 0
frames = []
results = []
Hs = []

# For real time (though it is slow)
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    frames.append(frame)
    
    if not ret: break
    
    if frame_count < start_frame:
        # cv.imshow('Processed Frame', cv.resize(frame, (0, 0), fx=0.5, fy=0.5))  # Display original frame
        out.write(frame)
        # if cv.waitKey(10) & 0xFF == ord('q'): break
        Hs.append(None)
        results.append(None)
        
    else:
        # find the keypoints and descriptors with SIFT
        kp2, des2 = sift.detectAndCompute(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), None)

        # BFMatcher with default params
        # bf = cv.BFMatcher()
        # matches = bf.knnMatch(des1,des2,k=2)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # Higher values will be slower but more accurate

        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Choose only the best matches
        all_good = []
        for m,n in matches:
            if m.distance < .75*n.distance:
                all_good.append([m])
                
        all_good.sort(key=lambda x: x[0].distance)  # Sort matches by distance
        top_n = 500  # Limit the number of matches to top N
        good = all_good[:top_n]  # Keep only the top N matches
                

        # Get the coordinates of the good matches in both images
        points1 = np.float32([kp1[m[0].queryIdx].pt for m in good])  # Keypoints in img1
        points2 = np.float32([kp2[m[0].trainIdx].pt for m in good])  # Keypoints in img2

        # Find the homography matrix
        try: H, mask = cv.findHomography(points1, points2, cv.RANSAC, 5.0)
        except:
            Hs.append(None) 
            results.append(None)
            # print("Missing one at", frame_count)
            frame_count += 1
            continue
        
        if previous_H is not None and H is not None:
            diff = np.linalg.norm(H - previous_H)  # Calculate the difference in matrices
            if diff > 75 and failed_count < 4:  # If the difference is too large, don't use the frame
                # H = previous_H  # Use the previous homography
                Hs.append(None) 
                results.append(None)
                # print("Missing one at", frame_count)
                frame_count += 1
                continue
            # print(diff)
        
        # Define the four corners of the reference image (cropped)
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        # Apply the homography to the corners of the reference painting
        
        try: corners_transformed = cv.perspectiveTransform(corners, H)
        except: 
            Hs.append(None)
            results.append(None)
            print("Missing one at", frame_count)
            frame_count += 1
            continue
        
        # Don't use the frame if the contour doesn't make sense
        if cv.isContourConvex(np.int32(corners_transformed)) == False or np.abs(H[2, 2]) > 1.5 or np.abs(H[2, 2]) < 0.5:
            H = previous_H
            Hs.append(None) 
            results.append(None)
            print("Missing one at", frame_count)
            continue
        previous_H = H  # Update the previous homography
        
        # Warp the reference image (img3_color) to fit the detected region in the video frame
        warped_reference = cv.warpPerspective(img3_color, H, (frame_width, frame_height))

        # Create a mask from the warped reference image (to blend it only in the transformed region)
        mask = np.zeros_like(frame)
        cv.fillConvexPoly(mask, np.int32(corners_transformed), (255, 255, 255))
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        # Use the mask to blend
        result = cv.bitwise_and(frame, frame, mask=255 - mask)
        result += cv.bitwise_and(warped_reference, warped_reference, mask=mask)

        # Write the processed frame to the new video file and display it
        out.write(result)
        cv.imshow('Processed Frame', cv.resize(result, (0, 0), fx=.5, fy=.5))
        if cv.waitKey(1) & 0xFF == ord('q'): break  # Press 'q' to exit the loop early
        
        Hs.append(H)
        results.append(result)
        
        # if frame_count > start_frame + 20: break   # TODO: Remove
    
    frame_count += 1

# Release the video objects
cap.release()
out.release()
cv.destroyAllWindows()


missing = [i for i, H in enumerate(Hs) if H is None and i > start_frame]
# print("missing", missing)

def fill_missing_homographies(H_list, missing):
    """Interpolates missing homographies in the list."""
    
    def interpolate_homography(H1, H2, alpha):
        """Linearly interpolates between two homography matrices."""
        return (1 - alpha) * H1 + alpha * H2
    
    for start_idx in missing:
        # Find the next valid homography
        next_valid_idx = None
        for j in range(start_idx + 1, len(H_list)):
            if H_list[j] is not None:
                next_valid_idx = j
                break
        
        # Interpolate only if valid homographies exist on both sides
        if start_idx > 0 and next_valid_idx is not None:
            H_start = H_list[start_idx - 1]
            H_end = H_list[next_valid_idx]

            # Fill the gap (exclude the valid endpoint itself)
            gap_size = next_valid_idx - start_idx
            for k in range(gap_size):
                alpha = (k + 1) / (gap_size + 1)
                H_list[start_idx + k] = interpolate_homography(H_start, H_end, alpha).astype(np.float32)

    return H_list

Hs = fill_missing_homographies(Hs, missing)

cap = cv.VideoCapture('video.MOV')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
# Do the same thing, but now not live, interpolating the spots where we failed before
out = cv.VideoWriter('processed_video_smoothed.MOV', cv.VideoWriter_fourcc(*'mp4v'), fps, 
                    (frame_width, frame_height))


for i, (H, frame, result) in enumerate(zip(Hs, frames, results)):
    if i < start_frame:
        # cv.imshow('Processed Frame', cv.resize(frame, (0, 0), fx=0.5, fy=0.5))  # Display original frame
        out.write(frame)
        # if cv.waitKey(10) & 0xFF == ord('q'): break
    else:
        if result is not None:  # Use original result if it already worked
            out.write(result)
        else:
            corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
            # Apply the homography to the corners of the reference painting
            corners_transformed = cv.perspectiveTransform(corners, H)
            warped_reference = cv.warpPerspective(img3_color, H, (frame_width, frame_height))

            # Create a mask from the warped reference image (to blend it only in the transformed region)
            mask = np.zeros_like(frame)
            cv.fillConvexPoly(mask, np.int32(corners_transformed), (255, 255, 255))
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

            # Use the mask to blend
            result = cv.bitwise_and(frame, frame, mask=255 - mask)
            result += cv.bitwise_and(warped_reference, warped_reference, mask=mask)
            
            out.write(result)
            
            
# Release the video objects
cap.release()
out.release()
cv.destroyAllWindows()