#Useful Utility function to complete tracking video.........
import os
import cv2
import csv
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm

# Initialize video capture and writer
def initialize_video(video_file, output_file):
    """
    Initializes video capture and writer objects for reading from the input video and writing to the output video.

    Args:
        video_file (str): Path to the input video file.
        output_file (str): Path to the output video file where the processed video will be saved.

    Returns:
        tuple:
            - video_cap (cv2.VideoCapture): The video capture object.
            - writer (cv2.VideoWriter): The video writer object for saving processed frames.
            - total_frames (int): Total number of frames in the input video.
            - frame_width (int): Width of each frame in the video.
            - frame_height (int): Height of each frame in the video.
    """
    video_cap = cv2.VideoCapture(video_file)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames for tqdm
    return video_cap, writer, total_frames, frame_width, frame_height


def extract_hsv(frame, ltrb):
    """
    Extracts the mean hue, saturation, and value (HSV) from the region of interest (ROI) defined by the bounding box.

    Args:
        frame (numpy.ndarray): The current video frame.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2) defining the ROI.

    Returns:
        tuple:
            - mean_hue (float): Mean hue value in the ROI.
            - mean_saturation (float): Mean saturation value in the ROI.
            - mean_value (float): Mean value (brightness) in the ROI.
    """
    x1, y1, x2, y2 = map(int, ltrb)
    # Ensure the bounding box coordinates are within the frame boundaries
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

    if x2 > x1 and y2 > y1:
        # Extract the region of interest (ROI) based on the bounding box
        object_roi = frame[y1:y2, x1:x2]

        # Convert the ROI to HSV color space
        hsv_roi = cv2.cvtColor(object_roi, cv2.COLOR_BGR2HSV)

        # Compute the mean values for hue, saturation, and value in the HSV space
        mean_hue = np.mean(hsv_roi[:, :, 0])
        mean_saturation = np.mean(hsv_roi[:, :, 1])
        mean_value = np.mean(hsv_roi[:, :, 2])

        return mean_hue, mean_saturation, mean_value
    return None, None, None

# Write tracking data to CSV
def write_csv(writer_csv, track_id, class_name, frame_count, bbox, hue, saturation, value):
    """
    Writes the tracking data for an object to the CSV file, including track ID, class, frame number, bounding box, and HSV color information.

    Args:
        writer_csv (csv.writer): CSV writer object to save tracking data.
        track_id (int): Unique ID for the tracked object.
        class_name (str): Name of the detected object class.
        frame_count (int): Current frame number.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        hue (float): Mean hue value of the object in the ROI.
        saturation (float): Mean saturation value of the object in the ROI.
        value (float): Mean value (brightness) of the object in the ROI.
    """
    writer_csv.writerow([track_id, class_name, frame_count, bbox, hue, saturation, value])
import csv
import cv2
import numpy as np
from tqdm import tqdm

def calculate_frame_difference(prev_frame, curr_frame):
    """
    Calculate the difference between two frames using Mean Squared Error (MSE).

    Args:
        prev_frame (np.array): The previous frame.
        curr_frame (np.array): The current frame.

    Returns:
        float: The difference between the two frames.
    """
    # Convert frames to grayscale for simplicity
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute Mean Squared Error (MSE) between the two frames
    diff = np.sum((prev_gray.astype("float") - curr_gray.astype("float")) ** 2)
    diff /= float(prev_gray.shape[0] * prev_gray.shape[1])

    return diff

