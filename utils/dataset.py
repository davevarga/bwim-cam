import os

import cv2
import numpy as np

from utils.augment import augment, augment_list, decode_augment
from utils.labels import read_labels, write_labels
from utils.plot import read_images


def get_corresponding(file_path, directory):
    """
    Given a file path (either an image or a label), returns the corresponding path in the opposite directory if it exists.

    Parameters:
        file_path (str): Path to the file (image or label).
        directory (str): Path to the corresponding file directory.

    Returns:
        str or None: Path to the corresponding file (label for an image, or image for a label), or None if not found.
    """
    file_name, ext = os.path.splitext(os.path.basename(file_path))

    # Determine if the file is an image or label based on the extension
    if ext.lower() in ['.jpg', '.png', '.jpeg']:
        # It's an image file, search for the corresponding label
        corresponding_path = os.path.join(directory, f"{file_name}.txt")
        return corresponding_path if os.path.exists(corresponding_path) else None

    elif ext.lower() == '.txt':
        # It's a label file, search for the corresponding image
        for image_ext in ['.jpg', '.png', '.jpeg']:
            potential_image_path = os.path.join(directory, f"{file_name}{image_ext}")
            if os.path.exists(potential_image_path):
                return potential_image_path

    # It does not have a corresponding image or Unsupported file type
    return None


def clean_unmatched(reference_dir, target_dir):
    """
    Deletes labels in the labels directory that don't have corresponding images.

    Parameters:
        target_dir (str): Directory where potentially unmatched files are located.
        reference_dir (str): Directory where corresponding file is potentially located

    Returns:
        None
    """
    counter = 0
    for file in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file)
        if os.path.isfile(file_path):
            corresponding_file_path = get_corresponding(file_path, reference_dir)
            if corresponding_file_path is None:
                os.remove(file_path)
                counter += 1

    print(f"Removed {counter} unmatched files from {target_dir}")


def remove_empty_labels(labels_dir):
    """
    Deletes all empty label files in the specified labels directory.
    Parameters:
        labels_dir (str): Path to the labels' directory.
    Returns:
        None
    """
    removed = 0
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        if os.path.isfile(label_path) and os.path.getsize(label_path) == 0:
            os.remove(label_path)
            removed += 1
    print(f"Removed {removed} empty label files from {labels_dir}")


def delete_unmatched(source_dir, reference_dir):
    """
    Deletes files in the source directory that are not present in the reference directory.

    Parameters:
        source_dir (str): Path to the source directory where files will be deleted.
        reference_dir (str): Path to the reference directory used for comparison.

    Returns:
        None
    """
    counter = 0

    # Create a set of filenames in the reference directory
    reference_files = set(os.listdir(reference_dir))

    # Iterate through files in the source directory
    for filename in os.listdir(source_dir):
        source_file_path = os.path.join(source_dir, filename)

        # Check if the file exists in the reference directory
        if filename not in reference_files:
            # If it doesn't exist, delete it
            os.remove(source_file_path)
            counter += 1

    print(f"Removed {counter} files from {source_dir}")


def dataset_augment(images_dir, labels_dir):
    images_path = sorted(os.listdir(images_dir))
    labels_path = sorted(os.listdir(labels_dir))

    # Use progress bar
    from tqdm import tqdm

    for image_path, label_path in tqdm(zip(images_path, labels_path), total=len(images_path), desc='Augmentation'):
        # Construct full path
        image_path = os.path.join(images_dir, image_path)
        label_path = os.path.join(labels_dir, label_path)

        # Load image into memory
        image = read_images(image_path)
        label = read_labels(label_path)

        for function in augment_list.keys():
            # Decode and apply the respective augmentation
            augment_function = decode_augment(function)
            augmented_image, augmented_label = augment(
                image, label, augment_function, apply_prob=0.5)

            # Check if the augmentation was applied (image differs from the original)
            if not np.array_equal(image, augmented_image):
                # Construct new file names with the augmentation type appended
                basename = os.path.basename(image_path)
                name, ext = os.path.splitext(basename)

                # Extract code for respective augmentation
                code = augment_list[function]

                # Construct from full path
                augmented_image_path = os.path.join(
                    images_dir,
                    f"{name}{code}{ext}")
                augmented_label_path = os.path.join(
                    labels_dir,
                    f"{name}{code}.txt")

                # Save the augmented image and label
                cv2.imwrite(augmented_image_path, augmented_image)
                write_labels(augmented_label_path, augmented_label)


if __name__ == "__main__":
    images_dir = '/home/davevarga/Projects/datasets/traffic-detection/train/images'
    labels_dir = '/home/davevarga/Projects/datasets/traffic-detection/train/labels'

    dataset_augment(images_dir, labels_dir)