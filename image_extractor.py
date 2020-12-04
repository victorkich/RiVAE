from cv2 import VideoCapture, imwrite
from tqdm import tqdm
import os

path = os.path.abspath(os.path.dirname(__file__))
images = f"{path}/data/images"
videos = f"{path}/data/videos"
listdir = os.listdir(videos)

# parameter
interval = 1

for rep in listdir:
    cap = VideoCapture(videos+'/'+rep)

    if not cap.isOpened():
        print("Error opening video stream or file")
        continue

    print(f"Extracting frames from video {rep}")
    for i in tqdm(range(0, int(cap.get(7)), interval)):
        ret, frame = cap.read()
        if ret:
            imwrite(f"{images}/{int(i/interval)}.jpg", frame)
        else:
            break

    cap.release()
