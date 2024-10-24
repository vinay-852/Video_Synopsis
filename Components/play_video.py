#Video Format which can be played in Colab..............
from IPython.display import HTML
from base64 import b64encode
import subprocess
def play_video(filename):
  html = ''
  video = open(filename,'rb').read()
  src = 'data:video/mp4;base64,' + b64encode(video).decode()
  html += fr'<video width=900 controls autoplay loop><source src="%s" type="video/mp4"></video>' % src
  return HTML(html)
def process_and_play_video(input_filename, output_filename):
  """
  Uses the ffmpeg command to process a video file and play it in Colab.

  Args:
      input_filename (str): The path to the input video file.
      output_filename (str): The path to save the processed video file.

  Returns:
      HTML: An HTML element to display the processed video in Colab.
  """
  subprocess.run([
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', input_filename,
        '-vcodec', 'libx264',
        output_filename,
        '-y'
    ])

  return play_video(output_filename)