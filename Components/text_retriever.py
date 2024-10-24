#Copy data without downloading.......
import os
import requests
import tqdm

def text_retriever(url) -> list[str]:
  """
  Retrieves URLs from a text file.

  Args:
      url (str): The URL of the text file containing video URLs.

  Returns:
      list[str]: A list of video URLs extracted from the text file.

  Raises:
      Exception: If there's an error retrieving or parsing the text file.
  """

  try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for non-2xx status codes

    text = response.text
    urls = text.splitlines()  # Split text into lines, removing unnecessary spaces

    return urls
  except Exception as e:
    print(f"Error retrieving URLs: {e}")
    return []