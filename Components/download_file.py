#Download Dataset form internet.........
import os
import requests
from tqdm import tqdm

def download_file(url, save_dir):
  """
  Download a file from a given URL and save it to a specified directory.

  Args:
    url: The URL of the file to download.
    save_dir: The directory where the file should be saved.

  Returns:
    None
  """
  try:
      if not os.path.exists(save_dir):
          os.makedirs(save_dir)

      filename = url.split('/')[-1]
      file_path = os.path.join(save_dir, filename)

      response = requests.get(url, stream=True)
      total_size = int(response.headers.get('content-length', 0))

      with open(file_path, 'wb') as f:
          with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as progress_bar:
              for data in response.iter_content(chunk_size=1024):
                  if data:
                      progress_bar.update(len(data))
                      f.write(data)

      print(f"Download complete! File saved to: {file_path}")
  except Exception as e:
      print(f"Error downloading file: {e}")