import os
import cv2
import math
import random
from copy import deepcopy
import matplotlib.pyplot as plt


from utils.labels import read_label

# Predefined color list
predefined_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']

# Predefined color list code association
color_dict = {
        'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0), 'yellow': (0, 255, 255),
        'cyan': (255, 255, 0), 'magenta': (255, 0, 255), 'orange': (0, 165, 255), 'purple': (128, 0, 128)
}


def read_images(image_paths):
    """Loads images from a list of file paths."""
    if isinstance(image_paths, str): return cv2.imread(image_paths)
    else: return [cv2.imread(path) for path in image_paths if cv2.imread(path) is not None]


def read_labels(label_file_paths):
    """Reads labels from a list of file in the specified format."""
    if isinstance(label_file_paths, str): return read_label(label_file_paths)
    labels = []
    for label_file_path in label_file_paths:
        labels.append(read_label(label_file_path))
    return labels


def load(image_paths, label_paths):
    """Loads images and labels into the memory from the list of file paths."""
    # Check if parameters are valid
    if len(image_paths) != len(label_paths):
        raise ValueError('Number of images and labels paths do not match.')

    images = read_images(image_paths)
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    labels = read_labels(label_paths)

    # Return images and labels
    return images, labels


def save_images(images, image_names, target_directory):
    """
    Saves a list of images to a specified directory using the provided image names.

    Parameters:
        images (list): List of images (as numpy arrays or any format supported by OpenCV).
        image_names (list): List of names for the saved images (must match the length of images).
        target_directory (str): Path to the target directory where images will be saved.

    Returns:
        None
    """
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    # Iterate through images and names
    for img, name in zip(images, image_names):
        # Construct the full path for the new image
        image_path = os.path.join(target_directory, name)

        # Save the image using OpenCV
        # Assuming img is in a format compatible with OpenCV
        cv2.imwrite(image_path, img)


def boxes(image, labels, class_names=None):
    """
    Plots bounding boxes around detected objects in an image.
d
    Parameters:
    - image: The image (numpy array) to plot the detections on.
    - labels: A list of labels where each label is in the format
              [class_id, x_center, y_center, width, height].
              Coordinates are normalized (range 0 to 1).
    - class_names: Optional list of class names corresponding to class_ids.

    Returns:
    - None (the function displays the image with plotted bounding boxes).
    """
    h, w, _ = image.shape

    # Convert normalized coordinates to actual pixel values
    for label in labels:
        print(label)
        class_id, x_center, y_center, box_width, box_height = label

        # Convert normalized center and size to absolute pixel values
        x_center_abs = int(x_center * w)
        y_center_abs = int(y_center * h)
        box_width_abs = int(box_width * w)
        box_height_abs = int(box_height * h)

        # Calculate top-left and bottom-right corners of the bounding box
        x1 = int(x_center_abs - box_width_abs / 2)
        y1 = int(y_center_abs - box_height_abs / 2)
        x2 = int(x_center_abs + box_width_abs / 2)
        y2 = int(y_center_abs + box_height_abs / 2)

        # Draw the bounding box on the image
        color = predefined_colors[class_id]
        color_code = color_dict[color]
        cv2.rectangle(image, (x1, y1), (x2, y2), color_code, 2)

        # Optionally, add a label with the class name
        if class_names:
            label_text = class_names[class_id] if class_id < len(class_names) else str(class_id)
            cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_code, 1)

    # Return image with bouding boxes
    return image


def show(image):
    """
    Plots a single image using matplotlib
    """
    plt.imshow(image)
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def grid(images, layout=None, figsize=(10, 10)):
    """
     Plots a list of images in a grid layout. If no layout is specified,
    automatically determines the best layout. Raises an exception if no images are provided.

    Parameters:
    - images (list of np.array): List of images (in RGB format) to be plotted.
    - layout (tuple): Optional. (rows, cols) specifying the grid layout.
    - figsize (tuple): Size of the entire figure.

    Returns:
    - None. Displays the images in a grid.
    """

    num_images = len(images)

    # Determine the best layout if not provided
    if layout is None:
        rows = math.isqrt(num_images)  # Find the largest integer square root
        cols = math.ceil(num_images / rows)
    else:
        rows, cols = layout

    # Create the plot with the specified or calculated layout
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array in case of multi-row layout

    # Plot each image in the grid
    for idx, img in enumerate(images):
        axes[idx].imshow(img)
        axes[idx].axis('off')  # Hide axes for each image

    # Hide any remaining axes if there are fewer images than grid cells
    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout(pad=0.2)
    plt.show()


def sample(dataset_path, num_images, labels_path=None, image_ext=".jpg", label_ext=".txt"):
    """
    Randomly selects a given number of images from a dataset and returns the list of image file paths.
    If a labels path is specified, it also returns the list of corresponding label files.

    Parameters:
    - dataset_path (str): Path to the directory containing image files.
    - num_images (int): The number of random images to select.
    - labels_path (str): Optional. Path to the directory containing label files.
                         Labels must have the same base name as the images, but with a different extension.
    - image_ext (str): The file extension of the images. Default is ".jpg".
    - label_ext (str): The file extension of the labels . Default is ".txt".

    Returns:
    - images (list): List of paths to the selected image files.
    - labels (list): If `labels_path` is specified, returns the corresponding list of label file paths.
                     Otherwise, returns None.

    Raises:
    - ValueError: If the number of images to select is greater than the number of available images in the dataset.
    """

    # Get all image files from the dataset directory
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(image_ext)]

    # Check if the requested number of images is greater than 0
    if num_images <= 0:
        raise ValueError(f"Number of images to select is less or equal to 0: {num_images}")

    # Check if the requested number of images is available
    if num_images > len(image_files):
        raise ValueError(f"Number of images to select ({num_images}) is "
                         f" than the number of available images ({len(image_files)}).")

    # Randomly select the given number of images
    selected_images = random.sample(image_files, num_images)

    # Get the full image paths
    images = [os.path.join(dataset_path, img) for img in selected_images]

    # Handle labels if the labels_path is provided
    label_ext = ".txt" if label_ext is None else label_ext
    if labels_path:
        labels = [os.path.join(labels_path, image.replace(image_ext, label_ext)) for image in selected_images]
        return images, labels
    else:
        return images


def convert_color(image, format):
    if format == 'bgr2rbg': return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif format == 'rgb2bgr': return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif format == 'bgr2bgra': return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif format == 'bgra2rbg': return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)


def frame(images, labels, class_names=None):
    """
    Frames labeled object in each image.

    Parameters:
    - images (list): Path to the directory containing image files.
    - labels (list): Path to the directory containing label files.
    - class_names (list): Name referring to objects in the image

    Returns:
    - The list of framed images.
    """

    # If only one image is provided
    if images.ndim == 3:
        return boxes(images, labels, class_names)

    # Frame detection object on every image, to validate object types and correct label annotation
    framed_images = []
    for image, label in zip(images, labels):
        image_with_detections = boxes(image, label, class_names)
        framed_images.append(image_with_detections)

    # Return framed images
    return framed_images


def random_plot(images_dir, labels_dir, num_of_images=4, class_names=None):
    """
    Select randomly a number of images from the path specified, and the correspong labels.
    After sampling from the dataset, frames objects labeled and plots it in a grid layout.
    The layout is determined by the amount of images to select, and aspect ratio.

    Parameters:
    - images_dir (str): Path to the directory containing image files.
    - labels_dir (str): Path to the directory containing label files.
    - class_names (list): Name referring to objects in the image
    - num_of_images (int): The number of random images to select.

    Returns:
    - Images and labels that where randomly selected for plotting.
     Displays the images in a grid.
    """
    # Check if label path and image path exists.
    assert os.path.isdir(images_dir), "Images directory does not exist."
    assert os.path.isdir(labels_dir), "Labels directory does not exist."

    # Sample 4 random image and label paths
    images_path, labels_path = sample(images_dir, num_of_images, labels_dir)

    # Read images and labels in memory
    images, labels = load(images_path, labels_path)

    # Preserve unframed images
    original_images = deepcopy(images)

    # Plot framed images in a grid
    framed_images = frame(images, labels, class_names)
    grid(framed_images, layout=None, figsize=(16, 9))

    # Return the images and labels path to be used again
    return original_images, labels


if __name__ == "__main__":
    images_dir = './datasets/uc-detrac/train/images'
    labels_dir = './datasets/uc-detrac/train/labels'
    names = ['bicycle', 'bus', 'car', 'motorbike', 'person', 'truck']

    # Chose target directory where images and labels will be saved
    random_plot(images_dir, labels_dir, num_of_images=4, class_names=names)

