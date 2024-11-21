import os
from utils.augment import center_crop
from utils.labels import write_labels
from utils.plot import load, save_images


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

if __name__ == "__main__":
    # Remove unmatched from val dataset
    images_dir = './datasets/uc-detrac/val/images'
    labels_dir = './datasets/uc-detrac/val/labels'

    # Configure croping parameters
    crop_percentage = 0.9
    crop_threshold = 0.2

    # Crop images
    image_paths = os.listdir(images_dir)
    label_paths = os.listdir(labels_dir)

    for image_path, label_path in zip(image_paths, label_paths):
        image_path = os.path.join(images_dir, image_path)
        label_path = os.path.join(labels_dir, label_path)

        # Load image and labels into memory
        [image], [labels] = load([image_path], [label_path])

        # Crop images
        cropped_image, cropped_labels = center_crop(image, labels, crop_percentage, crop_threshold)

        # Save cropped images
        name = os.path.basename(image_path)
        save_images([cropped_image], [name], images_dir)

        # Save modified labels
        write_labels(label_path, cropped_labels)