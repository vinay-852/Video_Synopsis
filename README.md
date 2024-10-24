# Video Processing and Analysis Toolkit

This repository provides a comprehensive toolkit designed for efficient video processing and analysis. The toolkit includes features such as background generation, object detection, video synopsis, and more, making it ideal for research, surveillance, or general video analysis projects.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Generate Background](#generate-background)
  - [Download Dataset](#download-dataset)
  - [Play Video](#play-video)
  - [Preprocess Video](#preprocess-video)
  - [Generate Video Synopsis](#generate-video-synopsis)
  - [Calculate Area and Velocity](#calculate-area-and-velocity)
  - [Extract Object Tracks](#extract-object-tracks)
  - [Optimize Tracks](#optimize-tracks)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with the toolkit, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/vinay-852/Video_Synopsis.git
   cd Video_Synopsis
   ```

2. Install the Requirements:
    ```bash
    pip install -r requirements.txt
    ```
3. Generate Background for static background Video:
    ```python
    from Components.generate_background import generate_background

    generate_background('input_video.mp4', 'background.jpg', method='median', num_frames=30)
    ```

## Requirements

The following packages are required to run the toolkit:

```markdown
supervision
git+https://github.com/THU-MIG/yolov10.git
deep-sort-realtime
opencv-python-headless
absl-py
torch
tqdm
requests
ipython
```

## Download Dataset

Download a dataset of video files using the text_retriever and download_file functions in text_retriever.py and download_file.py:

```python
from Components.text_retriever import text_retriever
from Components.download_file import download_file

urls = text_retriever("https://raw.githubusercontent.com/Kitware/MEVID/refs/heads/main/mevid-v1-video-URLS.txt")

for url in urls:
    download_file(url, "dataset")
```

## Preprocess Video

Initialize video preprocessing using the initialize_video function in preprocessing.py:

```python
from Components.preprocessing import initialize_video

video_cap, writer, total_frames, frame_width, frame_height = initialize_video('input_video.mp4', 'output_video.mp4')
```

## Extract Object Tracks

Extract object tracks from a video using the extract_object_tracks function:

```python
from extractObjectTracks import extract_object_tracks

extract_object_tracks('model_file.pth', 'input_video.mp4', 'output_tracks.csv', conf_threshold=0.5, threshold=0.3)
```

## Calculate Area and Velocity

Calculate the area and velocity of object tracks using the calculate_area_velocity_tubes function:

```python
from calculate_area_velocity_tubes import calculate_area_velocity_tubes

input_csv = 'object_tracks.csv'
output_csv = 'object_tracks_with_area_velocity_tubes.csv'

calculate_area_velocity_tubes(input_csv, output_csv)
```

## Optimize Tracks

Optimize object tracks using the simulated_annealing function:

```python
from optimise import simulated_annealing

optimized_tracks = simulated_annealing(database, initial_temperature=1000, cooling_rate=0.995, mutation_rate=0.1, max_generations=100)
```

## Generate Video Synopsis

Generate a video synopsis using the generate_video_synopsis function:

```python
from videoSynopsis import generate_video_synopsis

generate_video_synopsis('background.jpg', 'optimized_person_tracks.csv', 'crops_dir', 'synopsis_frames', video_output='synopsis_video.mp4', fps=30, max_objects_per_frame=4)
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

Feel free to customize the [README.md](README.md) file as needed.