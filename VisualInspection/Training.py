import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt 
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from matplotlib import cm

GOOD = 0
BAD = 1
UGLY = 2

def load_images_from_folder(folder_path):
    images = []  # List to store loaded images
    file_names = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img = cv.imread(file_path)  # Load the image using OpenCV
            if img is not None:  # Ensure the image was loaded successfully
                images.append(img)
                file_names.append(filename)
            else:
                print(f"Warning: Failed to load {file_path}")
    return images, file_names

def calculate_red_proportion(image, lower_red1=(0, 50, 50), lower_red2=(170, 50, 50), 
                            upper_red1=(10, 255, 255), upper_red2=(180, 255, 255)):
    """
    Calculates the proportion of red pixels in an image.
    
    Args:
        image (numpy.ndarray): The input image in HSV color space
        lower_red (tuple): Lower bound for red in HSV.
        upper_red (tuple): Upper bound for red in HSV.
            In HSV (Red in HSV typically spans two ranges)
                1. Low range (0-10 H): e.g., (0, 50, 50) to (10, 255, 255)
                2. High range (170-180 H): e.g., (170, 50, 50) to (180, 255, 255)

    Returns:
        float: The proportion of red pixels in the image.
    """
    red_mask1 = cv.inRange(image, lower_red1, upper_red1)   # Create a binary mask for red pixels
    red_mask2 = cv.inRange(image, lower_red2, upper_red2)   # Create a binary mask for red pixels
    red_pixel_count = cv.countNonZero(red_mask1)  # Calculate the number of red pixels
    red_pixel_count += cv.countNonZero(red_mask2)
    total_pixels = image.shape[0] * image.shape[1]  # Calculate the total number of pixels in the image
    red_proportion = red_pixel_count / total_pixels  # Calculate the proportion of red pixels
    return red_proportion

def calculate_white_proportion(image, lower_white=(0, 0, 200), upper_white=(180, 55, 255)):
    """
    Calculates the proportion of white pixels in an image.
    
    Args:
        image (numpy.ndarray): The input image in HSV color space
        lower_white (tuple): Lower bound for white in HSV.
        upper_white (tuple): Upper bound for white in HSV.
            In HSV:
                - Low saturation (S) and high value (V) define white.

    Returns:
        float: The proportion of white pixels in the image.
    """
    white_mask = cv.inRange(image, lower_white, upper_white)  # Create a binary mask for white pixels
    white_pixel_count = cv.countNonZero(white_mask)  # Calculate the number of white pixels
    total_pixels = image.shape[0] * image.shape[1]  # Calculate the total number of pixels in the image
    white_proportion = white_pixel_count / total_pixels  # Calculate the proportion of white pixels
    return white_proportion

def calculate_green_proportion(image, lower_green=(35, 50, 50), upper_green=(85, 255, 255)):
    """
    Calculates the proportion of green pixels in an image.
    
    Args:
        image (numpy.ndarray): The input image in HSV color space
        lower_green (tuple): Lower bound for green in HSV.
        upper_green (tuple): Upper bound for green in HSV.
            In HSV (Green typically spans around 60°-180° H):
                - Green: e.g., (35, 50, 50) to (85, 255, 255)
    
    Returns:
        float: The proportion of green pixels in the image.
    """
    green_mask = cv.inRange(image, lower_green, upper_green)  # Create a binary mask for green pixels
    green_pixel_count = cv.countNonZero(green_mask)  # Calculate the number of green pixels
    total_pixels = image.shape[0] * image.shape[1]  # Calculate the total number of pixels in the image
    green_proportion = green_pixel_count / total_pixels  # Calculate the proportion of green pixels
    return green_proportion

def calculate_yellow_proportion(image, lower_yellow=(20, 100, 100), upper_yellow=(40, 255, 255)):
    """
    Calculates the proportion of yellow pixels in an image.
    
    Args:
        image (numpy.ndarray): The input image in HSV color space
        lower_yellow (tuple): Lower bound for yellow in HSV.
        upper_yellow (tuple): Upper bound for yellow in HSV.
            In HSV (Yellow typically spans around 20°-40° H):
                - Yellow: e.g., (20, 100, 100) to (40, 255, 255)
    
    Returns:
        float: The proportion of yellow pixels in the image.
    """
    yellow_mask = cv.inRange(image, lower_yellow, upper_yellow)  # Create a binary mask for yellow pixels
    yellow_pixel_count = cv.countNonZero(yellow_mask)  # Calculate the number of yellow pixels
    total_pixels = image.shape[0] * image.shape[1]  # Calculate the total number of pixels in the image
    yellow_proportion = yellow_pixel_count / total_pixels  # Calculate the proportion of yellow pixels
    return yellow_proportion

def calculate_black_gray_proportion(image, lower_gray=(0, 0, 50), upper_gray=(180, 50, 150), lower_black=(0, 0, 0), upper_black=(180, 255, 50)):
    """
    Calculates the proportion of black or gray pixels in an image.
    
    Args:
        image (numpy.ndarray): The input image in HSV color space
        lower_gray (tuple): Lower bound for gray in HSV.
        upper_gray (tuple): Upper bound for gray in HSV.
        lower_black (tuple): Lower bound for black in HSV.
        upper_black (tuple): Upper bound for black in HSV.
            In HSV:
                - Black pixels typically have very low Value (V close to 0).
                - Gray pixels have low Saturation (S) and middle-to-low Value (V).
    
    Returns:
        float: The proportion of black and gray pixels in the image.
    """
    gray_mask = cv.inRange(image, lower_gray, upper_gray)  # Create a binary mask for gray pixels
    black_mask = cv.inRange(image, lower_black, upper_black)  # Create a binary mask for black pixels
    
    gray_pixel_count = cv.countNonZero(gray_mask)  # Calculate the number of gray pixels
    black_pixel_count = cv.countNonZero(black_mask)  # Calculate the number of black pixels
    
    total_pixels = image.shape[0] * image.shape[1]  # Calculate the total number of pixels in the image
    black_gray_proportion = (gray_pixel_count + black_pixel_count) / total_pixels  # Calculate the proportion of black and gray pixels
    return black_gray_proportion

def compute_lbp(arr):
    """Find LBP of all pixels.
    Also perform Vectorization/Normalization to get feature vector.
    """
    # LBP function params
    radius = 3
    n_points = 8 * radius
    METHOD = 'uniform'
    n_bins = n_points + 2
    
    lbp = local_binary_pattern(arr, n_points, radius, METHOD)
    lbp = lbp.ravel()
    # feature_len = int(lbp.max() + 1)
    feature = np.zeros(n_bins)
    for i in lbp:
        feature[int(i)] += 1
    feature /= np.linalg.norm(feature, ord=1)
    return feature

def calculate_area_perimeter(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    area = 0
    perimeter = 0
    for contour in contours:
        area += cv.contourArea(contour)
        perimeter += cv.arcLength(contour, True)
    return area, perimeter

def calculate_circularity(area, perimeter):
    if perimeter == 0:
        return 0
    return 4 * np.pi * area / (perimeter ** 2)

def calculate_edge_intensity(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray_image, 100, 200)
    edge_intensity = np.sum(edges) / (image.shape[0] * image.shape[1])
    return edge_intensity

def calculate_area_perimeter(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    area = 0
    perimeter = 0
    for contour in contours:
        area += cv.contourArea(contour)
        perimeter += cv.arcLength(contour, True)
    return area, perimeter

def calculate_circularity(area, perimeter):
    if perimeter == 0:
        return 0
    return 4 * np.pi * area / (perimeter ** 2)

def calculate_edge_intensity(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray_image, 100, 200)
    edge_intensity = np.sum(edges) / (image.shape[0] * image.shape[1])
    return edge_intensity

def calculate_fourier_descriptors(image, num_coefficients=10):
    """Calculate the Fourier Descriptors of an object's contour in the image.
    
    Args:
        image (numpy.ndarray): The input image in BGR or grayscale format.
        num_coefficients (int): The number of Fourier coefficients to use.
        
    Returns:
        np.ndarray: A feature vector containing the Fourier Descriptors.
    """
    # Convert the image to grayscale if it's not already
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Threshold to get the binary object
    _, binary_image = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is the object of interest
    try: contour = max(contours, key=cv.contourArea)
    except: return np.zeros(num_coefficients)
    
    # Convert the contour points to complex numbers (x + iy)
    contour_complex = np.array([complex(pt[0][0], pt[0][1]) for pt in contour], dtype=complex)
    
    # Perform the Fourier Transform
    fourier_coefficients = np.fft.fft(contour_complex)
    
    # Keep the first 'num_coefficients' coefficients (including the zero frequency term)
    fourier_coefficients = fourier_coefficients[:num_coefficients]
    
    # Normalize the descriptors to avoid scale issues
    fourier_coefficients = np.abs(fourier_coefficients)  # Magnitude of Fourier coefficients
    fourier_coefficients /= np.linalg.norm(fourier_coefficients, ord=1)  # Normalize
    
    if len(fourier_coefficients) != num_coefficients:
        while len(fourier_coefficients) < 10: 
            fourier_coefficients = np.concatenate([fourier_coefficients, [0]])
        
    return fourier_coefficients

def get_features(img):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # Convert to HSV
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # Convert to HSV
    
    # Color features
    red_proportion = calculate_red_proportion(hsv_image)
    white_proportion = calculate_white_proportion(hsv_image)
    green_proportion = calculate_green_proportion(hsv_image)
    yellow_proportion = calculate_yellow_proportion(hsv_image)
    black_proportion = calculate_black_gray_proportion(hsv_image)
    
    # Shape & Edge features
    area, perimeter = calculate_area_perimeter(img)
    circularity = calculate_circularity(area, perimeter)
    edge_intensity = calculate_edge_intensity(img)
    
    # Texture features (LBP)
    lbp_features = compute_lbp(hsv_image[:, :, 2])
    fourier_coefficients = calculate_fourier_descriptors(img)

    # Combine all features
    color_features = [red_proportion, white_proportion, green_proportion, yellow_proportion, black_proportion]
    shape_features = [area, perimeter, circularity, edge_intensity]
    features = np.concatenate((color_features, lbp_features, shape_features, fourier_coefficients), axis=0)

    # Print features
    # print(f"Image {i+1}:")
    # print(f"  Red Proportion: {red_proportion:.2%}, White Proportion: {white_proportion:.2%}")
    # print(f"  Green Proportion: {green_proportion:.2%}, Yellow Proportion: {yellow_proportion:.2%}")
    # print(f"  Black Proportion: {black_proportion:.2%}")
    # print(f"  Area: {area:.2f}, Perimeter: {perimeter:.2f}")
    # print(f"  Circularity: {circularity:.2f}, Edge Intensity: {edge_intensity:.2f}")
    
    return features

def get_all_features(good_images, bad_images, ugly_images):
    all_images = [good_images, bad_images, ugly_images]
    all_features = []

    for j, images in enumerate(all_images):
        image_features = []

        for i, img in enumerate(images):
            features = get_features(img)
            image_features.append(features)

        image_features = np.array(image_features)
        all_features.append(image_features)
        print("Finished loading dataset", j+1)

    return all_features

def shuffle_and_split(good_features, good_files, bad_features, bad_files, ugly_features, ugly_files, test_size=0.2, random_seed=42):
    """
    Shuffle the data and split into training and test sets.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, train_files, test_files)
            - X_train: Training feature vectors
            - X_test: Testing feature vectors
            - y_train: Training labels (if applicable)
            - y_test: Testing labels (if applicable)
            - train_files: Corresponding file names for training set
            - test_files: Corresponding file names for testing set
    """
    # Store all into one
    features = np.concatenate([good_features, bad_features, ugly_features])
    file_names = np.concatenate([good_files, bad_files, ugly_files])
    labels = np.concatenate([[GOOD]*len(good_features), [BAD]*len(bad_features), [UGLY]*len(ugly_features)])
    
    np.random.seed(random_seed)   # Set the random seed for reproducibility

    # Shuffle the indices
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    
    # Split the indices into training and test sets
    test_size_int = int(len(features) * test_size)
    train_indices = indices[:-test_size_int]
    test_indices = indices[-test_size_int:]
    
    # Split the data into features and filenames
    X_train = features[train_indices]
    X_test = features[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    train_files = [file_names[i] for i in train_indices]
    test_files = [file_names[i] for i in test_indices]
    
    return X_train, X_test, train_labels, test_labels, train_files, test_files

def get_conf_mat(y_pred, y_target, n_cats):
    """Build confusion matrix from scratch.
    (This part could be a good student assignment.)
    """
    conf_mat = np.zeros((n_cats, n_cats))
    n_samples = y_target.shape[0]
    for i in range(n_samples):
        _t = y_target[i]
        _p = y_pred[i]
        conf_mat[_t, _p] += 1
    norm = np.sum(conf_mat, axis=1, keepdims=True)
    return conf_mat / norm

def vis_conf_mat(conf_mat, cat_names, acc):
    """Visualize the confusion matrix and save the figure to disk."""
    n_cats = conf_mat.shape[0]

    fig, ax = plt.subplots()
    # figsize=(10, 10)

    cmap = cm.Blues
    im = ax.matshow(conf_mat, cmap=cmap)
    im.set_clim(0, 1)
    ax.set_xlim(-0.5, n_cats - 0.5)
    ax.set_ylim(-0.5, n_cats - 0.5)
    ax.set_xticks(np.arange(n_cats))
    ax.set_yticks(np.arange(n_cats))
    ax.set_xticklabels(cat_names)
    ax.set_yticklabels(cat_names)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=45,
            ha="right", rotation_mode="anchor")

    for i in range(n_cats):
        for j in range(n_cats):
            text = ax.text(j, i, round(
                conf_mat[i, j], 2), ha="center", va="center", color="w")

    cbar = fig.colorbar(im)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    _title = 'Normalized confusion matrix, acc={0:.2f}'.format(acc)
    ax.set_title(_title)

    # plt.show()
    _filename = 'conf_mat.png'
    plt.savefig(_filename, bbox_inches='tight')

if __name__=="__main__":
    # Load in the images
    good_images, good_files = load_images_from_folder("contours/Good1_Slow")
    bad_images, bad_files = load_images_from_folder("contours/Bad_Slow")
    ugly_images, ugly_files = load_images_from_folder("contours/Ugly_Slow")

    # Get the features
    all_features = get_all_features(good_images, bad_images, ugly_images)
    good_features, bad_features, ugly_features = all_features

    # Split into training and test set
    train_data, test_data, train_labels, test_labels, train_files, test_files = \
        shuffle_and_split(good_features, good_files, bad_features, bad_files, ugly_features, ugly_files)
        
    print("Start the training")
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(train_data, train_labels)            # fit SVM using training data
    prediction = clf.predict(test_data)          # make prediction on the test data

    # Save weights to a .npy file (using numpy)
    weights = clf.coef_
    intercept = clf.intercept_
    classes = clf.classes_
    np.savez('svc_model.npz', coef=weights, intercept=intercept, classes=classes)

    # visualization
    cmat = get_conf_mat(y_pred=prediction, y_target=test_labels, n_cats=3)
    acc = cmat.trace() / cmat.shape[0]
    vis_conf_mat(cmat, ["Good", "Bad", "Ugly"], acc)

    # cv.imshow("Sample Image", ugly_images[-1])  # Display the last image
    # cv.waitKey(0)  # Wait for a key press to close the window
    # cv.destroyAllWindows()