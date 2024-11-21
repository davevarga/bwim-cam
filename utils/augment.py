import os
import numpy as np

import utils.labels
from utils.plot import frame


def center_crop(image, labels=None, percentage=0.9, threshold=0.2):
    """
    Crops an image to its center by a specified percentage and adjusts object labels accordingly.

    Parameters:
    - image (np.ndarray): The input image as a NumPy array (height, width, channels).
    - labels (list of tuples): List of object labels in the format (object_id, center_x, center_y, width, height).
       Each label uses normalized coordinates (0 to 1) relative to the original image size.
    - percentage (float): The percentage (0 < percentage <= 100) of the original image dimensions to retain in the crop.
    - threshold (float, optional): Minimum proportion of an object's area (from the original bounding box)
        that must be visible in the cropped image to retain the label. Default is 0.1 (10% visibility).

    Returns:
    - cropped_image (np.ndarray): The cropped image as a NumPy array.
    - adjusted_labels (list of tuples): List of adjusted labels for the cropped image in the same format as input labels.
        Coordinates and dimensions are rounded to six decimal places.

    Usage:
    This function is ideal for preprocessing images in object detection tasks, where it's necessary to
    center-crop images while ensuring object labels remain accurate. Labels for objects mostly outside
    the cropped area are filtered out based on the threshold parameter, allowing control over whether
    partially visible objects should be retained.
    """
    # Handle arguments expressed as int types
    # This is due to division operation used while assessing abs new area
    if type(percentage) == int:
        percentage = float(percentage) / 100

    if type(percentage) == int:
        threshold = float(threshold) / 100

    # Ensure the percentage is within a reasonable range
    assert 0 < percentage <= 1.0, "Percentage must be between 0% and 100%."
    assert 0 < threshold <= 1.0, "Threshold must be between 0% and 100%."

    # Get original dimensions
    orig_height, orig_width = image.shape[:2]

    # Calculate new dimensions based on the center crop
    new_width = int(orig_width * percentage)
    new_height = int(orig_height * percentage)

    # Calculate the cropping coordinates
    start_x = (orig_width - new_width) // 2
    start_y = (orig_height - new_height) // 2
    end_x = start_x + new_width
    end_y = start_y + new_height

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]

    # Return without labels
    if labels is None: return cropped_image

    # Adjust labels to fit the cropped image
    adjusted_labels = []
    for label in labels:
        object_id, center_x, center_y, width, height = label

        # Convert normalized coordinates to absolute coordinates for adjustment
        abs_center_x = center_x * orig_width
        abs_center_y = center_y * orig_height
        abs_width = width * orig_width
        abs_height = height * orig_height

        # Calculate the boundaries of the label box
        box_x_min = abs_center_x - abs_width / 2
        box_x_max = abs_center_x + abs_width / 2
        box_y_min = abs_center_y - abs_height / 2
        box_y_max = abs_center_y + abs_height / 2

        # Check if any part of the box is within the crop boundaries
        if (box_x_max > start_x and box_x_min < end_x and
                box_y_max > start_y and box_y_min < end_y):

            # Calculate the intersection area within the crop
            intersect_x_min = max(box_x_min, start_x)
            intersect_x_max = min(box_x_max, end_x)
            intersect_y_min = max(box_y_min, start_y)
            intersect_y_max = min(box_y_max, end_y)

            # Calculate are of new bounding box object
            new_area = max(0, intersect_x_max - intersect_x_min) * max(0, intersect_y_max - intersect_y_min)

            # Normalize new area with cropping amount for accurate comparison
            normalized_new_area = new_area * percentage

            # Calculate the original bounding box area
            original_area = abs_width * abs_height

            # Calculate the proportion of the original area that is within the cropped area
            area_proportion = normalized_new_area / original_area

            # If the visible portion is below the threshold, skip this label
            if area_proportion < threshold:
                continue

            # Adjust box coordinates relative to the cropped area
            new_center_x = (intersect_x_min + intersect_x_max) / 2 - start_x
            new_center_y = (intersect_y_min + intersect_y_max) / 2 - start_y
            new_label_width = (intersect_x_max - intersect_x_min)
            new_label_height = (intersect_y_max - intersect_y_min)

            # Normalize the new coordinates with respect to the cropped image dimensions
            new_center_x = round(new_center_x / new_width, 6)
            new_center_y = round(new_center_y / new_height, 6)
            new_label_width = round(new_label_width / new_width, 6)
            new_label_height = round(new_label_height / new_height, 6)

            # Append adjusted label
            adjusted_labels.append(
                [int(object_id), new_center_x, new_center_y,
                 new_label_width, new_label_height])

    return cropped_image, adjusted_labels


def crop(image, labels, crop):
    """
    Crops an image to the specified coordinates and adjusts bounding box labels accordingly.
     Bounding boxes outside the crop are removed, and partially visible ones are updated to reflect their visible portion.

    Parameters:
    - image (ndarray): A 2D (grayscale) or 3D (color) NumPy array representing the image to be cropped.
        Example shape: (H, W, C) or (H, W).
    - labels (ndarray or list): A list or NumPy array of bounding box labels in YOLO format, where each row is:
        [class_id, center_x, center_y, width, height] (all values normalized between 0 and 1).
        Example: [[0, 0.5, 0.5, 0.4, 0.4], [1, 0.8, 0.2, 0.3, 0.3]]

    - crop_coords (tuple): A tuple defining the crop region as (x_min, y_min, x_max, y_max) in pixel coordinates.
        The function automatically adjusts these values to fit within the image dimensions.

    Returns:
    - cropped_image (ndarray): The cropped section of the input image.
    - adjusted_labels (list): Updated list of bounding box labels in YOLO format, adjusted for the cropped image.
        Bounding boxes are normalized relative to the cropped image size, rounded to 6 decimal places.

    Raises:
    - ValueError: Raised if:
        The image is not a valid NumPy array.
        Labels are not in the expected format or are empty.
        crop_coords is not a valid tuple of four integers.
        Crop dimensions are invalid (e.g., x_min >= x_max or y_min >= y_max).
    """
    if not isinstance(image, np.ndarray) or len(image.shape) not in [2, 3]:
        raise ValueError("Image must be a 2D or 3D numpy array.")

    if not isinstance(labels, (list, np.ndarray)) or len(labels) == 0:
        raise ValueError("Labels must be a non-empty list or numpy array.")

    if not isinstance(crop, tuple) or len(crop) != 4:
        raise ValueError("crop_coords must be a tuple of four integers (x_min, y_min, x_max, y_max).")

    x_min, y_min, x_max, y_max = crop

    # Ensure crop coordinates are within image boundaries
    height, width = image.shape[:2]
    x_min = max(0, min(x_min, width))
    y_min = max(0, min(y_min, height))
    x_max = max(0, min(x_max, width))
    y_max = max(0, min(y_max, height))

    # Ensure valid crop dimensions
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Invalid crop dimensions: Ensure x_min < x_max and y_min < y_max within image boundaries.")

    cropped_image = image[y_min:y_max, x_min:x_max]

    # Image dimensions before and after cropping
    orig_height, orig_width = image.shape[:2]
    crop_width, crop_height = x_max - x_min, y_max - y_min

    adjusted_labels = []
    for label in labels:
        class_id, cx, cy, w, h = label

        # Convert normalized YOLO format to pixel coordinates
        cx_abs, cy_abs = cx * orig_width, cy * orig_height
        w_abs, h_abs = w * orig_width, h * orig_height

        # Calculate bounding box edges
        x1, y1 = cx_abs - w_abs / 2, cy_abs - h_abs / 2
        x2, y2 = cx_abs + w_abs / 2, cy_abs + h_abs / 2

        # Adjust bounding box coordinates to the crop
        x1_new = max(x1 - x_min, 0)
        y1_new = max(y1 - y_min, 0)
        x2_new = min(x2 - x_min, crop_width)
        y2_new = min(y2 - y_min, crop_height)

        # Check if the bounding box is valid after cropping
        if x1_new < x2_new and y1_new < y2_new:
            # Calculate new center and dimensions in pixel coordinates
            new_cx = (x1_new + x2_new) / 2
            new_cy = (y1_new + y2_new) / 2
            new_w = x2_new - x1_new
            new_h = y2_new - y1_new

            # Normalize back to YOLO format
            new_cx_norm = new_cx / crop_width
            new_cy_norm = new_cy / crop_height
            new_w_norm = new_w / crop_width
            new_h_norm = new_h / crop_height

            # Round float values
            new_cx_norm = round(new_cx_norm, 8)
            new_cy_norm = round(new_cy_norm, 8)
            new_w_norm = round(new_w_norm, 8)
            new_h_norm = round(new_h_norm, 8)

            adjusted_labels.append([class_id, new_cx_norm, new_cy_norm, new_w_norm, new_h_norm])

    return cropped_image, adjusted_labels

if __name__ == '__main__':
    # Path to the image and labels dataset
    images_dir = './datasets/uc-detrac/val/images'
    labels_dir = './datasets/uc-detrac/val/labels'
    object_names = ['car', 'bus', 'van', 'other']

    # Target directory to export
    target_images_dir = './datasets/uc-detrac/temp/images'
    target_labels_dir = './datasets/uc-detrac/temp/labels'

    # Configure crop amount and threshold
    crop_percentage = 0.9
    crop_threshold = 0.2

    # List all images that are already in target dir
    images_paths = os.listdir(target_images_dir)

    # Iterate over all images in the directory
    for image_path in images_paths:

        # Find corresponding label file
        image_path = os.path.join(target_images_dir, image_path)
        label_path = utils.labels.correspond(image_path, labels_dir)

        # If label path does not exist then skit this cycle
        if label_path is None: continue

        # Swapping framed image with clean image from source
        clean_image_basename = os.path.basename(image_path)
        clean_image_path = os.path.join(images_dir, clean_image_basename)

        # Read image and labels into memory
        [image], [label] = utils.plot.load([clean_image_path], [label_path])

        # Crop image and label
        cropped_image, cropped_label = center_crop(
            image, label, crop_percentage, crop_threshold)

        # Verify if labels were filtered out
        print(f'Image {clean_image_path} cropped.'
              f' Number of objects: {len(label)} --> {len(cropped_label)}')

        # Frame cropped images with bounding boxes
        cropped_image = frame(cropped_image, cropped_label, object_names)

        # Write cropped image and labels to the destination
        name = os.path.basename(image_path)
        utils.plot.save_images(
            [cropped_image], [name], target_images_dir)
        print(f'Image {image_path} saved in {target_images_dir}')

        # Configure output label path
        label_basename = os.path.basename(label_path)
        output_label_path = os.path.join(target_labels_dir, label_basename)

        utils.labels.write_labels(output_label_path, cropped_label)
        print(f'Labels {output_label_path} saved in {target_labels_dir}')