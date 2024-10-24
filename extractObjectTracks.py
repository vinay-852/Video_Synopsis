from Components.preprocessing import *
from Components.yoloObjectDetection import *


def extract_object_tracks(model_file, video_file, output_file, conf_threshold, threshold):
    """
    Main function to extract object tracks from key frames of a video using YOLO for object detection and
    DeepSort for tracking. The function saves the tracking data, including bounding boxes and HSV values,
    to a CSV file and generates an output video with annotated tracks.

    Args:
        model_file (str): Path to the YOLO model file.
        video_file (str): Path to the input video file.
        output_file (str): Path to the output video file to save the annotated video.
        conf_threshold (float): Confidence threshold to filter weak detections.
        threshold (float): Threshold for selecting key frames based on frame difference.
    """
    try:
        # Initialize video capture, writer, and other video properties
        video_cap, writer, total_frames, frame_width, frame_height = initialize_video(video_file, output_file)

        # Load YOLO model and COCO class names
        model, class_names = load_yolo_model(model_file)

        # Initialize DeepSort tracker
        tracker = DeepSort(max_age=20, n_init=3)

        # Open CSV file for saving object tracks
        with open('object_tracks.csv', mode='w', newline='') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow(['Track ID', 'Class Name', 'Frame', 'Bounding Box', 'Hue', 'Saturation', 'Value'])

            frame_count = 0
            prev_frame = None

            # Progress bar for processing video frames
            with tqdm(total=total_frames, desc="Processing Frames") as pbar:
                while True:
                    ret, frame = video_cap.read()
                    frame_count += 1
                    if not ret:
                        print("End of video.")
                        break

                    # If this is the first frame, store it and continue to the next
                    if prev_frame is None:
                        prev_frame = frame
                        continue

                    # Calculate the difference between the current frame and the previous frame
                    frame_diff = calculate_frame_difference(prev_frame, frame)

                    # Check if the difference exceeds the threshold (key frame selection)
                    if frame_diff > threshold:
                        print(f"Processing key frame {frame_count} (difference: {frame_diff})")

                        # Object detection
                        detections = detect_objects(model, frame, conf_threshold)
                        tracks = update_tracker(tracker, detections, frame)

                        for track in tracks:
                            if track.is_confirmed():
                                track_id = track.track_id
                                ltrb = track.to_ltrb()  # Bounding box format: (left, top, right, bottom)
                                class_id = track.get_det_class()
                                class_name = class_names[class_id]

                                # Extract HSV color features from the bounding box region
                                hue, saturation, value = extract_hsv(frame, ltrb)

                                if hue is not None:
                                    # Write the tracking data (including HSV) to the CSV file
                                    write_csv(writer_csv, track_id, class_name, frame_count, ltrb, hue, saturation, value)

                                    # Draw bounding box and label on the frame
                                    x1, y1, x2, y2 = map(int, ltrb)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, f'{class_name} {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Write the frame with annotations to the output video
                        writer.write(frame)

                    # Update the previous frame
                    prev_frame = frame
                    pbar.update(1)

        # Release video capture and writer resources
        video_cap.release()
        writer.release()
        print("Tracking completed and saved to 'object_tracks.csv'.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage of the function with keyframe selection
model_file = "weights/yolov10l.pt"
video_file = "dataset/2018-05-16.14-25-01.14-30-01.school.G639.r13.avi"
output_file = "output_tracked_keyframes.mp4"
conf_threshold = 0.5
frame_diff_threshold = 5  # Set a threshold for keyframe selection

# Call the main function to extract object tracks based on key frames
extract_object_tracks(model_file, video_file, output_file, conf_threshold, frame_diff_threshold)
