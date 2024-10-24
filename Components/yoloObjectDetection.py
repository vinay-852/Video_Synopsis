#YOLOv10 model Operations..........
from typing import List
import torch
from text_retriever import text_retriever
from ultralytics import YOLOv10

# Load YOLO model and COCO classes
def load_yolo_model(model_file, class_file="coco.names"):
    """
    Loads the YOLO model for object detection and the class names from the COCO dataset.

    Args:
        model_file (str): Path to the YOLO model file.
        class_file (str): Path to the file containing COCO class names.

    Returns:
        tuple:
            - model (YOLOv10): The loaded YOLO model.
            - class_names (list): List of class names corresponding to COCO dataset classes.
    """
    # Check if a GPU is available; if not, use the CPU
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using device: {device}')

    # Load the YOLO model
    model = YOLOv10(model_file)

    # Load COCO class names from the specified file
    class_names=text_retriever("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")

    return model, class_names

def detect_objects(model, frame, conf_threshold):
    """
    Performs object detection on a given frame using the YOLO model.

    Args:
        model (YOLOv10): The loaded YOLO model used for detecting objects.
        frame (numpy.ndarray): The input video frame to perform object detection on.
        conf_threshold (float): Confidence threshold to filter weak detections.

    Returns:
        list: A list of detections where each detection contains bounding box, confidence, and class ID.
    """
    # Perform object detection on the frame
    results = model(frame, verbose=False)[0]

    detections = []
    for det in results.boxes:
        confidence = det.conf
        label = det.cls
        bbox = det.xyxy[0]  # Bounding box coordinates: [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        # Filter out weak detections based on confidence threshold
        if confidence >= conf_threshold:
            detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    return detections