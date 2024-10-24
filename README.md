# Video Processing and Analysis Toolkit for Surveillance
---
This repository provides a comprehensive toolkit designed for efficient video processing and analysis, specifically tailored for surveillance applications. The toolkit includes features such as background generation, object detection, video synopsis, and more, making it ideal for research, security, and monitoring projects.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Generate Background](#generate-background)
  - [Download MEVID Dataset](#download-mevid-dataset)
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

To set up the surveillance toolkit, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/vinay-852/Video_Synopsis.git
   cd Video_Synopsis
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate Background for surveillance video:
   ```python
   from Components.generate_background import generate_background

   generate_background('surveillance_video.mp4', 'background.jpg', method='median', num_frames=30)
   ```

## Requirements

The following Python packages are needed for the surveillance video toolkit:

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

## Download MEVID Dataset

To download the **MEVID** dataset, use the `text_retriever` and `download_file` functions:

```python
from Components.text_retriever import text_retriever
from Components.download_file import download_file

urls = text_retriever("https://raw.githubusercontent.com/Kitware/MEVID/refs/heads/main/mevid-v1-video-URLS.txt")

for url in urls:
    download_file(url, "dataset")
```

## Preprocess Surveillance Video

Preprocess the video for analysis using the `initialize_video` function:

```python
from Components.preprocessing import initialize_video

video_cap, writer, total_frames, frame_width, frame_height = initialize_video('surveillance_video.mp4', 'output_video.mp4')
```

## Extract Object Tracks from Surveillance Video

Extract tracks for moving objects (people, vehicles, etc.) using `extract_object_tracks`:

```python
from extractObjectTracks import extract_object_tracks

extract_object_tracks('model_file.pth', 'surveillance_video.mp4', 'output_tracks.csv', conf_threshold=0.5, threshold=0.3)
```

## Calculate Area and Velocity of Objects

Calculate the area and velocity of moving objects from extracted tracks:

```python
from calculate_area_velocity_tubes import calculate_area_velocity_tubes

input_csv = 'object_tracks.csv'
output_csv = 'object_tracks_with_area_velocity.csv'

calculate_area_velocity_tubes(input_csv, output_csv)
```

## Optimize Object Tracks for Video Synopsis

Optimize the object tracks for generating an efficient video synopsis:

```python
from optimise import simulated_annealing

optimized_tracks = simulated_annealing(database, initial_temperature=1000, cooling_rate=0.995, mutation_rate=0.1, max_generations=100)
```

## Generate Video Synopsis for Surveillance Footage

Create a compressed video synopsis that highlights key activity from the surveillance video:

```python
from videoSynopsis import generate_video_synopsis

generate_video_synopsis('background.jpg', 'optimized_object_tracks.csv', 'crops_dir', 'synopsis_frames', video_output='synopsis_video.mp4', fps=30, max_objects_per_frame=4)
```

This function will create a time-compressed surveillance video that highlights key actions, enabling efficient review of long surveillance footage.

## Contributing

We welcome contributions! If you encounter bugs or have suggestions for improvements, feel free to submit a pull request or open an issue.
