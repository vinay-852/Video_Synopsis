import cv2
import numpy as np

def generate_background(video_file, output_file='background.jpg', method='median', num_frames=30):
    """
    Generate a background image from the first 'num_frames' frames of a video by averaging or taking the median.

    Args:
        video_file (str): Path to the input video file.
        output_file (str): Path where the background image will be saved.
        method (str): Method to generate background ('mean' or 'median').
        num_frames (int): Number of frames to use for background generation.

    Returns:
        None
    """
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Initialize a list to store frames
    frames = []
    frame_count = 0

    # Read frames from the video until num_frames are collected
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video reached at frame {frame_count}.")
            break

        # Append the frame to the list
        frames.append(frame)
        frame_count += 1

    # Release the video capture object
    cap.release()

    # Convert the list of frames to a numpy array
    frames = np.array(frames)

    # Generate the background by computing the median or mean of the frames
    if method == 'median':
        background = np.median(frames, axis=0).astype(dtype=np.uint8)  # Median
    elif method == 'mean':
        background = np.mean(frames, axis=0).astype(dtype=np.uint8)  # Mean
    else:
        print("Error: Method must be 'mean' or 'median'.")
        return

    # Save the generated background as an image file
    cv2.imwrite(output_file, background)
    print(f"Background image saved as {output_file}")

# Example usage:
video_file = '/content/dataset/2018-05-16.14-25-01.14-30-01.school.G639.r13.avi'  # Replace with the path to your video file
generate_background(video_file, output_file='background.jpg', method='median', num_frames=30)
