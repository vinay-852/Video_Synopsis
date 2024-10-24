import numpy as np
import pandas as pd
import random
import math
from copy import deepcopy

# Load tracking data from CSV
def load_tracking_database(csv_file):
    """
    Load the tracking data from a CSV file into a list of dictionaries.

    Args:
        csv_file (str): Path to the CSV file containing tracking data.

    Returns:
        list: List of tracking records with bounding boxes and frame information.
    """
    df = pd.read_csv(csv_file)

    # Clean 'Bounding Box' column and convert to list of coordinates
    df['Bounding Box'] = df['Bounding Box'].apply(lambda x: [int(coord) for coord in x.strip('()').split(',')])

    # Filter only person class
    df = df[df['Class Name'] == 'person']

    # Convert dataframe to list of dictionaries
    database = df.to_dict(orient='records')
    return database

# Group the tracking data by Track ID
def group_tracks_by_id(database):
    """
    Group the tracking data by Track ID to create a list of tracks.

    Args:
        database (list): List of tracking records with bounding boxes and frame information.

    Returns:
        list: List of tracks with bounding boxes and frame information.
    """
    grouped_tracks = {}
    for record in database:
        track_id = record['Track ID']
        if track_id not in grouped_tracks:
            grouped_tracks[track_id] = []
        grouped_tracks[track_id].append(record)
    return list(grouped_tracks.values())

# Utility to compute Intersection over Union (IoU)
def iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (list): List of bounding box coordinates [x1, y1, x2, y2].
        box2 (list): List of bounding box coordinates [x1, y1, x2, y2].

    Returns:
        float: IoU value between the two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2

    # Calculate intersection
    inter_x1 = max(x1, x1_b)
    inter_y1 = max(y1, y1_b)
    inter_x2 = min(x2, x2_b)
    inter_y2 = min(y2, y2_b)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

# Temporal coherence: smoothness of bounding box motion over frames
def temporal_coherence(tracks):
    """
    Compute the temporal coherence score based on the smoothness of bounding box motion.

    Args:
        tracks (list): List of tracks with bounding boxes and frame information.

    Returns:
        float: Temporal coherence score based on bounding box displacement.
    """
    coherence_score = 0
    for track in tracks:
        for i in range(1, len(track)):
            box_prev = track[i-1]['Bounding Box']
            box_curr = track[i]['Bounding Box']
            displacement = np.linalg.norm(np.array(box_prev[:2]) - np.array(box_curr[:2]))
            coherence_score += np.exp(-displacement)
    return coherence_score

# Spatial coherence: minimize overlap between bounding boxes in the same frame
def spatial_coherence(tracks):
    """
    Compute the spatial coherence penalty for the bounding boxes in the same frame.
    
    Args:
        tracks (list): List of tracks with bounding boxes and frame information.
        
    Returns:
        float: Spatial coherence penalty based on bounding box overlap.
    """
    overlap_penalty = 0
    frames = {}
    for track in tracks:
        for frame_info in track:
            frame = frame_info['Frame']
            box = frame_info['Bounding Box']
            if frame not in frames:
                frames[frame] = []
            frames[frame].append(box)

    for boxes in frames.values():
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                overlap_penalty += max(0, iou(boxes[i], boxes[j]) - 0.1)  # Penalize for high overlap

    return overlap_penalty

# Fitness function with temporal and spatial coherence
def fitness(tracks, w1=1.5, w2=2):
    """
    Compute the fitness score of the tracks based on temporal and spatial coherence.
    
    Args:
        tracks (list): List of tracks with bounding boxes and frame information.
        w1 (float): Weight for temporal coherence.
        w2 (float): Weight for spatial coherence.
        
    Returns:
        float: Fitness score of the tracks.
    """
    return w1 * temporal_coherence(tracks) - w2 * spatial_coherence(tracks)

# Function to shift a bounding box by a certain amount to reduce overlap
def shift_bbox(bbox, shift_x, shift_y):
    """
    Shift the bounding box by a certain amount to reduce overlap with other bounding boxes.
    
    Args:
        bbox (list): List of bounding box coordinates [x1, y1, x2, y2].
        shift_x (int): Amount to shift in the x-direction.
        shift_y (int): Amount to shift in the y-direction.
        
    Returns:
        list: Adjusted bounding box coordinates after shifting.
    """
    x1, y1, x2, y2 = bbox
    return [x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y]

# Adjust BBoxes in a frame to minimize overlap
def adjust_frame_bboxes(frame_boxes):
    """
    Adjust the bounding boxes in a frame to minimize overlap between them.

    Args:
        frame_boxes (list): List of bounding boxes in the frame.

    Returns:
        list: Adjusted list of bounding boxes with no overlaps.
    """
    adjusted_boxes = deepcopy(frame_boxes)

    for i in range(len(adjusted_boxes)):
        for j in range(i + 1, len(adjusted_boxes)):
            while iou(adjusted_boxes[i], adjusted_boxes[j]) > 0:
                # Shift the second bounding box if overlap occurs
                adjusted_boxes[j] = shift_bbox(adjusted_boxes[j], shift_x=2, shift_y=2)

    return adjusted_boxes

# Mutate by shifting bounding boxes to avoid overlaps
def mutate(individual, mutation_rate=0.05):
    """
    Mutate the individual (list of tracks) by shifting bounding boxes to avoid overlaps.
    
    Args:
        individual (list): List of tracks with bounding boxes and frame information.
        mutation_rate (float): Probability of mutation for each track.

    Returns:
        list: Mutated list of tracks with adjusted bounding boxes.
    """
    mutated = deepcopy(individual)
    frames = {}

    for track in mutated:
        for obj in track:
            frame = obj['Frame']
            if frame not in frames:
                frames[frame] = []
            frames[frame].append(obj['Bounding Box'])

    # Adjust BBoxes in each frame to avoid overlap
    for frame, boxes in frames.items():
        frames[frame] = adjust_frame_bboxes(boxes)

    # Apply adjusted BBoxes back to tracks
    for track in mutated:
        for obj in track:
            frame = obj['Frame']
            obj['Bounding Box'] = frames[frame].pop(0)

    return mutated

# Simulated Annealing with track shifting and enhanced mutation
def simulated_annealing(database, initial_temperature=1000, cooling_rate=0.995, mutation_rate=0.1, max_generations=100):
    """
    Perform Simulated Annealing optimization on the tracking data to reduce overlap between bounding boxes.
    
    Args:
        database (list): List of tracks with bounding boxes and frame information.
        initial_temperature (float): Initial temperature for simulated annealing.
        cooling_rate (float): Rate at which the temperature is reduced.
        mutation_rate (float): Probability of mutation for each generation.
        max_generations (int): Maximum number of generations to run the optimization.

    Returns:
        list: Optimized list of tracks with adjusted bounding boxes.
    """
    # Initialize population with a single solution
    current_solution = deepcopy(database)

    current_fitness = fitness(current_solution)
    temperature = initial_temperature

    for generation in range(max_generations):
        # Mutate the current solution
        new_solution = mutate(current_solution, mutation_rate)
        new_fitness = fitness(new_solution)

        # Calculate the acceptance probability
        if new_fitness > current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
        else:
            acceptance_probability = math.exp((new_fitness - current_fitness) / temperature)
            if random.random() < acceptance_probability:
                current_solution = new_solution
                current_fitness = new_fitness

        # Cool down the temperature
        temperature *= cooling_rate
        if generation % 50 == 0 or generation == max_generations - 1:
            print(f"Generation {generation}, Fitness: {current_fitness:.4f}, Temperature: {temperature:.2f}")

    return current_solution

# Main script to load the CSV and run simulated annealing
if __name__ == "__main__":
    csv_file = 'object_tracks.csv'  # Replace with the path to your CSV file
    tracking_database = load_tracking_database(csv_file)

    # Group the tracks by Track ID
    grouped_tracks = group_tracks_by_id(tracking_database)

    # Run simulated annealing on the grouped tracks
    best_tracks = simulated_annealing(grouped_tracks)

    # Flatten the list of tracks for output
    optimized_data = []
    for track in best_tracks:
        optimized_data.extend(track)

    # Convert back to DataFrame for saving or further processing
    optimized_df = pd.DataFrame(optimized_data)
    optimized_df.to_csv('optimized_person_tracks.csv', index=False)

    print("Optimized Person Tracking Data Saved to 'optimized_person_tracks.csv'")
    print(optimized_df.head())
