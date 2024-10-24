import csv
import math

def calculate_area_velocity_tubes(input_csv, output_csv):
    """
    Calculate area and velocity for each object in the object_tracks.csv file by considering objects in the same tube
    (i.e., same Track ID) and computing the velocity based on the movement between consecutive frames.

    Args:
        input_csv (str): Path to the input CSV file (object_tracks.csv).
        output_csv (str): Path to the output CSV file to save the results with area and velocity.
    """
    try:
        rows = []

        # Read the input CSV file
        with open(input_csv, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert frame to int and bounding box values to float for calculations
                row['Frame'] = int(row['Frame'])
                # Parse the bounding box values by removing brackets and splitting by spaces
                row['Bounding Box'] = tuple(map(float, row['Bounding Box'].strip('[]').split()))
                rows.append(row)

        # Sort rows by Track ID and Frame to ensure sequential processing
        rows.sort(key=lambda x: (int(x['Track ID']), x['Frame']))

        # Open the output CSV file for writing
        with open(output_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write the new header with Area and Velocity columns
            writer.writerow(['Track ID', 'Class Name', 'Frame', 'Bounding Box', 'Hue', 'Saturation', 'Value', 'Area', 'Velocity'])

            prev_objects = {}  # To store the last seen object positions per Track ID

            # Loop through the rows and calculate area and velocity
            for row in rows:
                # Extract bounding box values
                left, top, right, bottom = row['Bounding Box']

                # Calculate area of the bounding box
                area = (right - left) * (bottom - top)

                # Calculate velocity for the same Track ID between consecutive frames
                track_id = row['Track ID']
                current_frame = row['Frame']

                # Calculate center of the current bounding box
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2

                # Check if we have a previous record for the same track_id (same tube)
                if track_id in prev_objects:
                    prev_frame, (prev_center_x, prev_center_y) = prev_objects[track_id]

                    # Calculate velocity if the object exists in the previous frame (adjacent frames only)
                    if current_frame == prev_frame + 1:
                        velocity = math.sqrt((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2)
                    else:
                        velocity = 0  # If not consecutive frames, velocity is 0
                else:
                    velocity = 0  # First appearance of the object, velocity is 0

                # Update the previous object record with the current frame and center
                prev_objects[track_id] = (current_frame, (center_x, center_y))

                # Write the row to the new CSV file with calculated area and velocity
                writer.writerow([
                    row['Track ID'], row['Class Name'], row['Frame'], row['Bounding Box'],
                    row['Hue'], row['Saturation'], row['Value'], area, velocity
                ])

        print(f"Area and velocity calculations completed and saved to {output_csv}.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Call the function with the input and output CSV file paths
input_csv = 'object_tracks.csv'
output_csv = 'object_tracks_with_area_velocity_tubes.csv'

calculate_area_velocity_tubes(input_csv, output_csv)