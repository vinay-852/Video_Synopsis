#Deepsort Operations...........
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

def initialize_deepsort(max_age=20, n_init=3):
    """
    Initializes the DeepSort tracker.

    Args:
        max_age (int): Maximum number of frames to keep alive a track.
        n_init (int): Minimum number of detections before a track is confirmed.

    Returns:
        DeepSort: An instance of the DeepSort tracker.
    """
    return DeepSort(max_age=max_age, n_init=n_init)

def update_tracker(tracker, detections, frame):
    """
    Updates the DeepSort tracker with new detections.

    Args:
        tracker (DeepSort): The DeepSort tracker instance.
        detections (list): List of detections in the format [bounding_box, confidence, class_id].
        frame (numpy.ndarray): The current video frame.

    Returns:
        List: Updated tracks from the DeepSort tracker.
    """
    return tracker.update_tracks(detections, frame=frame)
