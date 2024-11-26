import os
import shutil
from cProfile import label


def read_label(label_path):
    """
    This function reads YOLO format labels from a specified file.


    Each label is expected to be on a separate line with elements (class_id,
    center_x, center_y, width, height) separated by spaces. The function
    converts the class_id to an integer and the other elements to floats.

    :param label_path: The file path from which the labels will be read.
    :return: A list of labels, where each label is a list in the format
        [class_id (int), center_x (float), center_y (float),
        width (float), height (float)].
        If the file does not exist, an empty list is returned.
    """
    if not os.path.isfile(label_path):
        return []

    with open(label_path, 'r') as file:
        labels = [line.strip().split() for line in file.readlines()]

    # Convert label data to float
    return list([int(label[0]), float(label[1]), float(label[2]), float(label[3]), float(label[4])] for label in labels)


def read_labels(label_file_paths):
    """
    Reads labels from a list of label file paths.

    This function loads labels from one or more label files. If a single file path is provided,
    it returns the labels from that file. If a list of file paths is provided,
    it returns a list of labels for each file.

    :param label_file_paths: A single label file path as a string, or a list of label file paths.
    :return: The loaded labels. If a single file path is provided, the labels from that file are returned.
             If a list of file paths is provided, a list of labels for each file is returned.
    """
    if isinstance(label_file_paths, str): return read_label(label_file_paths)
    labels = []
    for label_file_path in label_file_paths:
        labels.append(read_label(label_file_path))
    return labels


def write_labels(label_path, labels):
    """
    This function writes YOLO format labels to a specified file.

    Each label is written on a new line, with the elements of the label
    (e.g., class_id, center_x, center_y, width, height) joined by a space.

    :param label_path: The file path where the labels will be written.
    :param labels: A list of labels, where each label is a tuple or list
        containing numeric values in YOLO format.
    :return: None
    """
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(" ".join(map(str, label)) + "\n")


def copy_files_to_directory(files, target_directory):
    """
    This function copies one or more files to a specified target directory.
    If the target directory does not exist, it will be created.
    The function handles both individual file paths and lists of file paths.
    It also skips invalid files and logs errors if file copying fails.

    :param files: A single file path as a string or a list of file paths to be copied.
    :param target_directory: The path of the directory where the files should be copied.
    :return: The number of files successfully copied to the target directory.
    """
    # If the input is not a list, make it a list
    num_files_copied = 0

    if not isinstance(files, list):
        files = [files]

    # Ensure the target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Copy each file to the target directory
    for file in files:
        if not os.path.isfile(file):
            print(f"Error: '{file}' is not a valid file.")
            continue

        # Construct the destination path
        destination = os.path.join(target_directory, os.path.basename(file))

        try:
            shutil.copy(file, destination)
            num_files_copied += 1
        except Exception as e:
            print(f"Error copying '{file}' to '{destination}': {e}")

    # How many files were copied successfully
    return num_files_copied


def delete_labels(labels, point, image_shape=None):
    """
    This function removes YOLO format labels that contain a given point.
    Each label is a bounding box in YOLO format, and the function checks
    if the specified point lies within any of these bounding boxes.
    Labels containing the point are removed from the list.

    :param labels: A list of YOLO labels, where each label is a tuple
        (class_id, center_x, center_y, width, height).
        Coordinates should be normalized unless `image_shape` is provided.
    :param point: A tuple (x, y) representing the coordinates of the point.
        If `image_shape` is provided, the coordinates are in pixels
        and will be normalized to YOLO format.
    :param image_shape: An optional tuple (height, width) or
        (height, width, channels) representing the shape of the image.
        If provided, the point's coordinates are normalized.
    :return: A list of YOLO labels that do not contain the specified point.
    """
    x, y = point

    # Normalize point if image_shape is provided
    # Works for grayscale images as well.
    height, width, _ = image_shape if len(image_shape) == 3 else (image_shape, 0)
    x /= width
    y /= height

    # Assert that point is in YOLO format range [0, 1]
    assert 0 <= x <= 1, "Point's x-coordinate must be in [0, 1] for normalized format."
    assert 0 <= y <= 1, "Point's y-coordinate must be in [0, 1] for normalized format."

    # Filter labels
    remaining_labels = []
    for label in labels:
        class_id, center_x, center_y, width, height = label
        x_min = center_x - width / 2
        x_max = center_x + width / 2
        y_min = center_y - height / 2
        y_max = center_y + height / 2

        # Check if the point is outside the label bounding box
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            remaining_labels.append(label)

    return remaining_labels


def delete_class(labels, target_class):
    """
    Changes the class ID of labels matching a specific class to a new class ID.

    :param labels: List of YOLO labels in the format (class_id, center_x, center_y, width, height).
    :param old_class: The class ID to be replaced.
    :param new_class: The new class ID that will replace the old class ID.
    :return: A list of YOLO labels with the specified class IDs updated.
    """
    return [label for label in labels if int(label[0]) != target_class]


def change_class(labels, old_class, new_class):
    """
    Changes the class ID of labels matching a specific class to a new class ID.

    Args:
        labels (list of tuples): List of YOLO labels in the format (class_id, center_x, center_y, width, height).
        old_class (int): The class ID to be changed.
        new_class (int): The new class ID to replace the old class ID.

    Returns:
        list of tuples: Updated list of YOLO labels with modified class IDs.
    """
    updated_labels = [
        [new_class, *label[1:]] if label[0] == old_class else label
        for label in labels
    ]
    return updated_labels


def permute_class(labels, permutation):
    """
    Permutes the class IDs of YOLO labels based on a specified permutation.

    :param labels: List of YOLO labels in the format (class_id, center_x, center_y, width, height).
    :param permutation: A list where the index represents the old class ID, and the value at that index is the new class ID.
        For example, a permutation of [2, 0, 1] means:
        - Class 0 becomes 2,
        - Class 1 becomes 0,
        - Class 2 becomes 1.
    :return: A list of YOLO labels with class IDs permuted according to the specified mapping.
    """

    updated_labels = [
        [permutation[label[0]], *label[1:]] if label[0] < len(permutation) else label
        for label in labels
    ]
    return updated_labels


def calculate(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    The function computes the area of intersection and union between
    two boxes represented by their center coordinates, width, and height.

    :param box1: A tuple (x_center, y_center, width, height) representing the first bounding box.
    :param box2: A tuple (x_center, y_center, width, height) representing the second bounding box.
    :return: A tuple containing the intersection area and the union area of the two bounding boxes.
    """

    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    # Calculate the intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Calculate area of intersection
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate area of each box
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    # Calculate the union area
    union_area = box1_area + box2_area - inter_area

    # Return intersection area and union area

    return inter_area, union_area


def correspond(image_paths, txt_directory=None):
    """
    This function matches image files with their corresponding label files.

    It expects the image files to have a `.jpg` extension, and it attempts
    to find a `.txt` label file with the same base name in the specified
    directory or in the same directory as the image file.

    :param image_paths: A single path or a list of image file paths in `.jpg` format.
    :param txt_directory: An optional directory where the corresponding `.txt` label files are stored.
                          If not provided, it uses the same directory as the image files.
    :return: A list of `.txt` label file paths corresponding to the input image files.
            If only one label file is found, it returns the path as a string.
            If no label files are found, it returns None.
    """

    # If the input is a single path, convert it into a list
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    txt_paths = []

    for image_file_path in image_paths:
        # Check if the input file is a .jpg file
        if not image_file_path.lower().endswith('.jpg'):
            raise ValueError(f"The input file {image_file_path} must be a .jpg file.")

        # Get the base name (without extension) of the image file
        base_name = os.path.splitext(os.path.basename(image_file_path))[0]

        # If no directory is specified, use the same directory as the image file
        if txt_directory is None:
            txt_directory = os.path.dirname(image_file_path)

        # Construct the full path for the .txt file
        txt_file_path = os.path.join(txt_directory, base_name + '.txt')

        # Check if the .txt file exists
        if not os.path.exists(txt_file_path):
            print(f"Error: The corresponding label file for '{image_file_path}' does not exist.")
        else: txt_paths.append(txt_file_path)

    # Function should return the same format as the parameter
    if len(txt_paths) == 0: return None
    else: return txt_paths if len(txt_paths) > 1 else txt_paths[0]


def remove_occluded_labels(objects, threshold=0.9):
    """
    This function removes labels of objects that are heavily occluded in an image.

    It compares each object in the input list with every other object to calculate
    the Intersection over Union (IoU) between their bounding boxes. If the occlusion
    (calculated as the intersection area over the smaller box area) exceeds the specified
    threshold, the smaller box is considered occluded and is removed.

    :param objects: A list of labels for objects in the image, where each label is in
                    the format (class_id, x_center, y_center, width, height).
    :param threshold: The IoU threshold above which objects are considered occluded.
                      Defaults to 0.9.

    :return: A list of labels with heavily occluded objects removed.
    """

    # List to keep track of indices to remove
    to_remove = set()

    # Compare each object with every other object
    for i in range(len(objects)-1):
        for j in range(i + 1, len(objects)):
            # (x_center, y_center, width, height)
            box1 = objects[i][1:]
            box2 = objects[j][1:]

            # Calculate box area
            area1 = box1[2] * box1[3]
            area2 = box2[2] * box2[3]

            # Calcuate occlusion
            inter_area, union_area = calculate(box1, box2)
            occlusion = inter_area / min(area1, area2)

            if occlusion > threshold:
                # Remove the smaller box
                if area1 < area2:
                    to_remove.add(i)
                else:
                    to_remove.add(j)

    # Remove labels of occluded objects
    filtered_labels = [label for idx, label in enumerate(objects) if idx not in to_remove]
    print(f"Processed and removed {len(to_remove)} occluded objects.")

    # Return the filtered labels back
    return filtered_labels


if __name__ == "__main__":
    images_dir = "/home/davevarga/Projects/datasets/traffic-surveillance/train/images"
    labels_dir = "/home/davevarga/Projects/datasets/traffic-surveillance/valid/labels"

    for label_path in os.listdir(images_dir):
        label_path = os.path.join(labels_dir, label_path)

        label = read_label(label_path)
        new_labels = delete_class(label, 2)
        print(new_labels)
        write_labels(label_path, new_labels)



