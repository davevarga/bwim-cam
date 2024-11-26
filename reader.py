import cv2
import pytesseract
from ultralytics import YOLO
from typing import List, Tuple
import numpy as np

YOUR_LICENSE_PLATE_ID = 0


class LicensePlateReader:
    def __init__(self, yolo_model_path: str, conf_threshold: float = 0.25, use_gpu: bool = True):
        """
        Initialize the YOLO model and other configurations.

        :param yolo_model_path: Path to the YOLO model file.
        :param conf_threshold: Confidence threshold for YOLO detections.
        :param use_gpu: Whether to use GPU acceleration for YOLO inference.
        """
        self.model = YOLO(yolo_model_path)  # Load YOLO model
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model.to(self.device)  # Use GPU if available

    def detect(self, image: np.ndarray) -> list[tuple[int, ...]]:
        """
        Detect license plates in the input image using the YOLO model.

        :param image: Input image as a numpy array.
        :return: List of bounding boxes [(x1, y1, x2, y2), ...].
        """
        results = self.model.predict(source=image, conf=self.conf_threshold, device=self.device)
        bboxes = []
        for result in results:
            for box in result.boxes.xyxy:  # YOLO detections as bounding boxes
                bboxes.append(tuple(map(int, box)))  # Convert coordinates to integers
        return bboxes

    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Preprocess the cropped license plate image for OCR.

        :param plate_image: Cropped license plate image as a numpy array.
        :return: Preprocessed image ready for OCR.
        """
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Binary thresholding
        return binary

    def ocr(self, plate_image: np.ndarray) -> str:
        """
        Perform OCR on the preprocessed license plate image.

        :param plate_image: Preprocessed license plate image.
        :return: Detected text.
        """
        text = pytesseract.image_to_string(plate_image, config='--psm 7')  # PSM 7: Single-line text
        return text.strip()

    def read(self, image_path: str) -> list[tuple[str, tuple[int, ...]]]:
        """
        Detect license plates in an image and read their text using OCR.

        :param image_path: Path to the input image.
        :return: List of tuples [(text, bbox), ...], where bbox = (x1, y1, x2, y2).
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image at {image_path}")

        # Detect license plates
        bboxes = self.detect(image)

        results = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            # Crop and preprocess the license plate image
            plate_image = image[y1:y2, x1:x2]
            preprocessed_image = self.preprocess(plate_image)

            # Perform OCR
            text = self.ocr(preprocessed_image)

            # Append the result
            results.append((text, bbox))

        return results

    def annotate(self, image_path: str, results: List[Tuple[str, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Annotate the image with detected license plate texts and bounding boxes.

        :param image_path: Path to the input image.
        :param results: List of tuples [(text, bbox), ...].
        :return: Annotated image as a numpy array.
        """
        image = cv2.imread(image_path)
        for text, (x1, y1, x2, y2) in results:
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put detected text above the bounding box
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image


# Demo tesseract
if __name__ == "__main__":
    # Model
    model = YOLO('license-plate/llp-detection5/weights/best.pt')
    results = model.predict(source='license.png', conf=0.25)

    # Load the image for cropping
    image = cv2.imread('license.png')

    # Iterate over detections
    for result in results:
        for box in result.boxes.xyxy:  # Get bounding boxes
            x1, y1, x2, y2 = map(int, box)  # Convert to integer pixel values

            # Crop the license plate from the image
            license_plate_image = image[y1:y2, x1:x2]

            # Preprocess for OCR
            gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Apply binary thresholding

            # Save or display the cropped license plate image (optional)
            cv2.imwrite('cropped_license_plate.jpg', binary)

            # OCR using Tesseract
            text = pytesseract.image_to_string(binary, config='--psm 7')
            print(f"Detected license plate text: {text.strip()}")

            # Optional: Draw bounding box and label on the original image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save or display the annotated image (optional)
    cv2.imwrite('annotated_image.jpg', image)
    cv2.imshow('Annotated Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()