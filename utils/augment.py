import os
import random

import cv2
import numpy as np

from utils import plot, labels

augment_list = {
    'horizontal_flip': 'hf', 'perspective_transform': 'pt', 'blur': 'b',
    'gaussian_noise': 'gn', 'salt_and_pepper': 'sp', 'brightness_contrast': 'bc',
    'color_jitter': 'cj', 'random_erasing': 're',
}

def decode_augment(function: str):
    # Geometric transformations
    if function == 'center_crop' or function == 'cc': return center_crop
    elif function == 'crop' or function == 'c': return crop
    elif function == 'horizontal_flip' or function == 'hf': return horizontal_flip
    elif function == 'perspective_transform' or function == 'pt': return perspective_transform

    # Occlusion and noise transformations
    elif function == 'blur' or function == 'b': return blur
    elif function == 'gaussian_noise' or function == 'gn': return gaussian_noise
    elif function == 'salt_and_pepper' or function == 'sp': return salt_and_pepper
    elif function == 'random_erasing' or function == 're': return random_erasing

    # Photometric transformations
    elif function == 'brightness_contrast' or function == 'bc': return brightness_contrast
    elif function == 'color_jitter' or function == 'cj': return color_jitter

    # If none of the above raise exception
    else: raise Exception(f"Augmentation function {function} not recognized")


def center_crop(image, labels=None, percentage=0.9, threshold=0.2):
    """
    Crops an image to its center and adjusts object labels.

    :param image: Input image as a NumPy array.
    :param labels: List of object labels (id, center_x, center_y, width, height).
    :param percentage: Percentage of original size to retain.
    :param threshold: Minimum object visibility in the crop.
    :return: Tuple of cropped image and adjusted labels.
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


def crop(image, labels, crop=None):
    """
    Crops an image and adjusts bounding box labels.

    :param image: Input image as a NumPy array.
    :param labels: List of bounding box labels in YOLO format.
    :param crop: Crop region as (x_min, y_min, x_max, y_max).
    :return: Tuple of cropped image and adjusted labels.
    :raises ValueError: If input parameters are invalid.
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


def horizontal_flip(image, label, apply_prob=0.5):
    """
    Horizontally flips an image and adjusts the YOLO-format label.

    :param image: Input image as a NumPy array.
    :param label: Label in YOLO format.
    :param apply_prob: Probability of applying the augmentation.
    :return: Tuple of flipped image and adjusted label.
    """
    if np.random.random() > apply_prob:
        # No augmentation; return original image and label
        return image, label

    # Flip the image along the Y-axis
    flipped_image = cv2.flip(image, 1)

    # Adjust labels
    adjusted_label = label.copy()
    for i in range(len(label)):
        class_id, center_x, center_y, width, height = label[i]
        # Flip the x-coordinate
        center_x = 1.0 - center_x
        adjusted_label[i] = [class_id, center_x, center_y, width, height]

    return flipped_image, adjusted_label


def perspective_transform(image, label, max_offset=0.2, apply_prob=0.5):
    """
    Applies a random perspective warp to an image and adjusts labels.

    :param image: Input image as a NumPy array.
    :param label: Labels in YOLO format.
    :param max_offset: Maximum offset for perspective warp.
    :param apply_prob: Probability of applying the augmentation.
    :return: Tuple of transformed image and adjusted labels.
    """
    if np.random.random() > apply_prob:
        # No augmentation; return original image and label
        return image, label

    # Get original dimensions
    original_h, original_w = image.shape[:2]

    # Generate random offsets for the corners
    dx1 = int(np.random.uniform(-max_offset, max_offset) * original_w)
    dy1 = int(np.random.uniform(-max_offset, max_offset) * original_h)
    dx2 = int(np.random.uniform(-max_offset, max_offset) * original_w)
    dy2 = int(np.random.uniform(-max_offset, max_offset) * original_h)
    dx3 = int(np.random.uniform(-max_offset, max_offset) * original_w)
    dy3 = int(np.random.uniform(-max_offset, max_offset) * original_h)
    dx4 = int(np.random.uniform(-max_offset, max_offset) * original_w)
    dy4 = int(np.random.uniform(-max_offset, max_offset) * original_h)

    # Define points for perspective transformation
    src_points = np.float32([
        [0, 0],
        [original_w, 0],
        [original_w, original_h],
        [0, original_h]
    ])
    dst_points = np.float32([
        [dx1, dy1],
        [original_w + dx2, dy2],
        [original_w + dx3, original_h + dy3],
        [dx4, original_h + dy4]
    ])

    # Compute perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective warp to the image
    transformed_image = cv2.warpPerspective(image, matrix, (original_w, original_h), flags=cv2.INTER_LINEAR)

    # Adjust labels
    adjusted_label = label.copy()
    for i in range(len(label)):
        class_id, center_x, center_y, width, height = label[i]

        # Convert normalized coordinates to absolute coordinates
        abs_center_x = center_x * original_w
        abs_center_y = center_y * original_h
        abs_width = width * original_w
        abs_height = height * original_h

        # Calculate corners of the bounding box in absolute coordinates
        box_corners = np.array([
            [abs_center_x - abs_width / 2, abs_center_y - abs_height / 2],
            [abs_center_x + abs_width / 2, abs_center_y - abs_height / 2],
            [abs_center_x + abs_width / 2, abs_center_y + abs_height / 2],
            [abs_center_x - abs_width / 2, abs_center_y + abs_height / 2]
        ], dtype=np.float32)

        # Apply the perspective transformation to the bounding box corners
        transformed_corners = cv2.perspectiveTransform(box_corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)

        # Get the new bounding box coordinates from transformed corners
        x_coords = transformed_corners[:, 0]
        y_coords = transformed_corners[:, 1]
        new_center_x = (x_coords.min() + x_coords.max()) / 2
        new_center_y = (y_coords.min() + y_coords.max()) / 2
        new_width = x_coords.max() - x_coords.min()
        new_height = y_coords.max() - y_coords.min()

        # Normalize the new bounding box coordinates
        center_x = new_center_x / original_w
        center_y = new_center_y / original_h
        width = new_width / original_w
        height = new_height / original_h

        # Clip values to keep them within [0, 1]
        center_x = np.clip(center_x, 0, 1)
        center_y = np.clip(center_y, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)

        adjusted_label[i] = [class_id, center_x, center_y, width, height]

    return transformed_image, adjusted_label


def blur(image, label, blur_kernel=5, apply_prob=0.5):
    """
    Applies Gaussian blur to an image.

    :param image: Input image as a NumPy array.
    :param label: Label in YOLO format.
    :param blur_kernel: Size of the Gaussian blur kernel. Must be odd.
    :param apply_prob: Probability of applying the augmentation.
    :return: Tuple of blurred image and unchanged label.
    """
    if np.random.random() > apply_prob:
        # No augmentation; return original image and label
        return image, label

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

    # Labels remain unchanged
    return blurred_image, label


def gaussian_noise(image, label, mean=0, stddev=10, apply_prob=0.5):
    """
    Adds Gaussian noise to an image.

    :param image: Input image as a NumPy array.
    :param label: Labels in YOLO format.
    :param mean: Mean of the Gaussian noise distribution.
    :param stddev: Standard deviation of the Gaussian noise distribution.
    :param apply_prob: Probability of applying the augmentation.
    :return: Tuple of noisy image and unchanged labels.
    """
    if np.random.random() > apply_prob:
        return image, label

    noise = np.random.normal(mean, stddev, image.shape).astype(np.int16)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image, label


def salt_and_pepper(image, label, salt_prob=0.015, pepper_prob=0.015, apply_prob=0.5):
    """
    Adds salt-and-pepper noise to an image.

    :param image: Input image as a NumPy array.
    :param label: Labels in YOLO format.
    :param salt_prob: Probability of salt noise.
    :param pepper_prob: Probability of pepper noise.
    :param apply_prob: Probability of applying the augmentation.
    :return: Tuple of noisy image and unchanged labels.
    """
    if np.random.random() > apply_prob:
        return image, label

    noisy_image = image.copy()
    total_pixels = image.shape[0] * image.shape[1]

    # Add salt noise
    num_salt = int(salt_prob * total_pixels)
    salt_coords = [np.random.randint(0, dim, num_salt) for dim in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1]] = [255, 255, 255]  # White pixels for salt noise

    # Add pepper noise
    num_pepper = int(pepper_prob * total_pixels)
    pepper_coords = [np.random.randint(0, dim, num_pepper) for dim in image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1]] = [0, 0, 0]  # Black pixels for pepper noise

    return noisy_image, label


def brightness_contrast(image, label, brightness_range=(0.25, 0.5), contrast_range=(0.75, 1.25), apply_prob=0.5):
    """
    Randomly adjusts brightness and contrast of an image.

    :param image: Input image as a NumPy array.
    :param label: Labels in YOLO format.
    :param brightness_range: Range for brightness adjustment.
    :param contrast_range: Range for contrast adjustment.
    :param apply_prob: Probability of applying the augmentation.
    :return: Tuple of adjusted image and unchanged labels.
    """
    if np.random.random() > apply_prob:
        return image, label

    brightness_factor = np.random.uniform(*brightness_range)
    contrast_factor = np.random.uniform(*contrast_range)

    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor * 100)
    return adjusted_image, label


def color_jitter(image, label, saturation_range=(0.5, 1.5), hue_shift_limit=20, apply_prob=0.5):
    """
    Randomly jitters the color of an image.

    :param image: Input image as a NumPy array.
    :param label: Labels in YOLO format.
    :param saturation_range: Range for saturation adjustment.
    :param hue_shift_limit: Maximum hue shift.
    :param apply_prob: Probability of applying the augmentation.
    :return: Tuple of color-jittered image and unchanged labels.
    """
    if np.random.random() > apply_prob:
        return image, label

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Adjust saturation
    saturation_factor = np.random.uniform(*saturation_range)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * saturation_factor, 0, 255)

    # Adjust hue
    hue_shift = np.random.uniform(-hue_shift_limit, hue_shift_limit)
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_shift) % 180

    jittered_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return jittered_image, label


def random_erasing(image, label, max_erase_ratio=0.4, fill_value=0, apply_prob=0.5):
    """
    Applies random erasing to objects in the image.

    :param image: Input image as a NumPy array.
    :param label: Labels in YOLO format.
    :param apply_prob: Probability of applying the augmentation.
    :param max_erase_ratio: Maximum proportion of an object to erase.
    :param fill_value: Value to fill the erased region.
    :return: Tuple of erased image and unchanged labels.
    """
    if np.random.random() > apply_prob:
        return image, label

    h, w, c = image.shape
    augmented_image = image.copy()

    for bbox in label:
        class_id, center_x, center_y, box_w, box_h = bbox

        # Calculate absolute bounding box coordinates
        box_abs_x1 = int((center_x - box_w / 2) * w)
        box_abs_y1 = int((center_y - box_h / 2) * h)
        box_abs_x2 = int((center_x + box_w / 2) * w)
        box_abs_y2 = int((center_y + box_h / 2) * h)

        box_width = box_abs_x2 - box_abs_x1
        box_height = box_abs_y2 - box_abs_y1

        # Ensure minimum valid erase dimensions (at least 1)
        min_erase_w = max(1, int(box_width * 0.1))  # Minimum 10% of box width
        min_erase_h = max(1, int(box_height * 0.1))  # Minimum 10% of box height

        # Calculate the maximum allowable erasure area
        max_erase_area = max_erase_ratio * box_width * box_height

        for _ in range(5):  # Attempt to find a valid erasure region
            erase_w = np.random.randint(min_erase_w, int(box_width * 0.5) + 1)
            erase_h = np.random.randint(min_erase_h, int(box_height * 0.5) + 1)

            if erase_w * erase_h > max_erase_area:
                continue

            erase_x = np.random.randint(box_abs_x1, box_abs_x2 - erase_w + 1)
            erase_y = np.random.randint(box_abs_y1, box_abs_y2 - erase_h + 1)

            # Ensure the erased region is within the bounding box
            erase_x1 = max(box_abs_x1, erase_x)
            erase_y1 = max(box_abs_y1, erase_y)
            erase_x2 = min(box_abs_x2, erase_x + erase_w)
            erase_y2 = min(box_abs_y2, erase_y + erase_h)

            # Apply erasure
            augmented_image[erase_y1:erase_y2, erase_x1:erase_x2, :] = fill_value
            break

    return augmented_image, label


def augment(image, label, function, **kwargs):
    """
    Applies an augmentation and saves the results.

    :param image: Path to the image.
    :param label: Path to the label.
    :param function: Augmentation function to apply.
    :param kwargs: Keyword arguments for the augmentation function.
    :return: With the augmented image and label if the image was augmented.
    :raise: If the function is not in the list of possible augmentation functions.
    """
    # If no augmentation is specified then abort
    if function.__name__ not in augment_list.keys():
        raise Exception('Augmentation function is not defined.')

    # Apply augmentation
    augmented_image, augmented_label = function(image, label, **kwargs)

    # Return augmentation solution
    return augmented_image, augmented_label


if __name__ == '__main__':
    # Example image and corresponding label
    image_path = '../datasets/uc-detrac/train/images/MVI_39761_img00056.jpg'
    label_path = '../datasets/uc-detrac/train/labels/MVI_39761_img00056.txt'

    # Load image into memory
    image = plot.read_images(image_path)
    label = labels.read_label(label_path)

    # Show initial object bounding boxes
    framed = plot.frame(image, label, ['car', '1', '2', '3'])
    plot.show(framed)

    # Apply augmentation
    augmented_image, augmented_label = augment(image, label, random_erasing, apply_prob=0.9)

    # Show augmented labels
    framed = plot.frame(augmented_image, augmented_label, ['car', '1', '2', '3'])
    plot.show(framed)
