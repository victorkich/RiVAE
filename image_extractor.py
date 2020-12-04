from cv2 import VideoCapture, imwrite
from tqdm import tqdm
import os

path = os.path.abspath(os.path.dirname(__file__))
images = f"{path}/data/images"
videos = f"{path}/data/videos"
listdir = os.listdir(videos)

# parameter
interval = 2

for rep in listdir:
    cap = VideoCapture(videos+'/'+rep)

    if not cap.isOpened():
        print("Error opening video stream or file")
        continue

    print(f"Calculating the number of frames from video {rep}")
    frames = 0
    while cap.isOpened():
        ret, _ = cap.read()
        if ret:
            frames += 1
        else:
            break

    print(f"Extracting frames from video {rep}")
    cap = VideoCapture(videos+'/'+rep)
    for i in tqdm(range(0, frames, interval)):
        ret, frame = cap.read()
        if ret:
            imwrite(f"{images}/{int(i/interval)}.jpg", frame)
        else:
            break

    cap.release()
