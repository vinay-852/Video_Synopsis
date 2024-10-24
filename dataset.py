from Components.text_retriever import text_retriever
from Components.download_file import download_file
urls=text_retriever("https://raw.githubusercontent.com/Kitware/MEVID/refs/heads/main/mevid-v1-video-URLS.txt")

for i in urls:
    download_file(i, "dataset")