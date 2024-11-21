import os
import shutil


def read_label(label_path):
    if not os.path.isfile(label_path):
        return []

    with open(label_path, 'r') as file:
        labels = [line.strip().split() for line in file.readlines()]

    # Convert label data to float
    return list([int(label[0]), float(label[1]), float(label[2]), float(label[3]), float(label[4])] for label in labels)


def write_labels(label_path, labels):
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(" ".join(map(str, label)) + "\n")


def copy_files_to_directory(files, target_directory):
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


def normalize_rect_coords(image_shape, rect_coords):
    """
    Normalizes rectangle coordinates to relative values.

    Args:
        image_shape: A tuple (height, width) representing the image dimensions.
        rect_coords: A tuple (x1, y1, x2, y2) representing the top-left and bottom-right corners of the rectangle.

    Returns:
        A tuple (center_x, center_y, width, height) representing the normalized rectangle parameters.
    """
    height, width, _ = image_shape
    x1, y1, x2, y2 = rect_coords

    # Calculate center coordinates
    center_x = (x1 + x2) / (2 * width)
    center_y = (y1 + y2) / (2 * height)

    # Calculate width and height relative to image size
    width = (x2 - x1) / width
    height = (y2 - y1) / height

    return center_x, center_y, width, height


def calculate(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Parameters:
        box1, box2 (tuple): Each box is represented by (x_center, y_center, width, height).
    Returns:
        float, float: intersection area and union area.
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
    Removes labels of objects that are heavily occluded in an image.

    Parameters:
        objects (str): Labels for the list of obejcts on the image
         (format: class_id, x_center, y_center, width, height).
        iou_threshold (float): Threshold above which objects are considered occluded.

    Returns:
        None
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