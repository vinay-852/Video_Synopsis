import cv2
import csv
import os
import numpy as np
from tqdm import tqdm

def generate_video_synopsis(bg_image_path, csv_file, crops_dir, output_dir, video_output=None, fps=30, max_objects_per_frame=5):
    """
    Generate a video synopsis by blending objects from different times into a single frame.

    Args:
        bg_image_path (str): Path to the background image for the video synopsis.
        csv_file (str): Path to the CSV file containing object tracking data.
        crops_dir (str): Directory containing cropped object images.
        output_dir (str): Directory to save the output frames.
        video_output (str): Path to save the output video file (optional).
        fps (int): Frames per second for the output video.
        max_objects_per_frame (int): Maximum number of objects to display in each frame.

    Returns:
        None
    """
    # Load the background image
    bg_image = cv2.imread(bg_image_path)
    if bg_image is None:
        raise ValueError(f"Background image {bg_image_path} not found.")

    # Get dimensions of the background
    frame_height, frame_width, _ = bg_image.shape

    # Initialize video writer if needed
    if video_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_output, fourcc, fps, (frame_width, frame_height))

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load tracking data from CSV file
    frame_dict = {}
    with open(csv_file, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            frame_num = int(row['Frame'])
            track_id = row['Track ID']
            class_name = row['Class Name']
            bbox = eval(row['Bounding Box'])  # Format: (x1, y1, x2, y2)
            crop_path = os.path.join(crops_dir, row['Crop Path'])

            # Organize data per frame
            if frame_num not in frame_dict:
                frame_dict[frame_num] = []
            frame_dict[frame_num].append((track_id, class_name, bbox, crop_path))

    # Object data structure: track ID -> list of frame info (bbox, crop path)
    track_data = {}
    for frame_num in sorted(frame_dict.keys()):
        for track_id, class_name, bbox, crop_path in frame_dict[frame_num]:
            if track_id not in track_data:
                track_data[track_id] = []
            track_data[track_id].append((frame_num, bbox, crop_path))

    # Step 2: Video synopsis by blending objects from different times
    synopsis_frames = []
    current_frame_num = 0
    used_tracks = set()

    while len(used_tracks) < len(track_data):
        # Create a new frame by cloning the background
        frame_image = bg_image.copy()

        # Select objects to display in this synopsis frame
        objects_in_frame = 0
        for track_id, track_frames in track_data.items():
            if track_id in used_tracks:
                continue

            # Find the next available frame for this track
            for frame_info in track_frames:
                frame_num, bbox, crop_path = frame_info
                if objects_in_frame >= max_objects_per_frame:
                    break  # Max number of objects reached for this frame

                # Blend object into the frame
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    continue  # Skip invalid bounding boxes

                # Load the cropped object image
                crop_img = cv2.imread(crop_path)
                if crop_img is None:
                    continue

                crop_height, crop_width, _ = crop_img.shape
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2

                top_left_x = int(centroid_x - (crop_width // 2))
                top_left_y = int(centroid_y - (crop_height // 2))

                top_left_x = max(0, min(top_left_x, frame_width - crop_width))
                top_left_y = max(0, min(top_left_y, frame_height - crop_height))

                try:
                    roi = frame_image[top_left_y:top_left_y + crop_height, top_left_x:top_left_x + crop_width]
                    crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(crop_gray, 1, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    fg_part = cv2.bitwise_and(crop_img, crop_img, mask=mask)
                    dst = cv2.add(bg_part, fg_part)
                    frame_image[top_left_y:top_left_y + crop_height, top_left_x:top_left_x + crop_width] = dst
                    objects_in_frame += 1
                    used_tracks.add(track_id)
                except Exception as e:
                    print(f"Error blending object in synopsis frame {current_frame_num}, track {track_id}: {str(e)}")


        # Optionally write the frame to the video
        if video_output:
            video_writer.write(frame_image)

        current_frame_num += 1


    # Release video writer if it was used
    if video_output:
        video_writer.release()

    print(f"Video synopsis complete. Frames saved to {output_dir}.")
    if video_output:
        print(f"Video saved to {video_output}.")

# Example usage
bg_image_path = "/content/background.jpg"
csv_file = "/content/optimized_person_tracks.csv"
crops_dir = "/content"
output_dir = "synopsis_frames"
video_output = "synopsis_video.mp4"  # Set to None if you don't need a video output

generate_video_synopsis(bg_image_path, csv_file, crops_dir, output_dir, video_output=video_output, fps=30, max_objects_per_frame=4)
