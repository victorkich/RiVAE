from tqdm import tqdm
import cv2
import os

path = os.path.abspath(os.path.dirname(__file__))
images = f"{path}/data/images"
videos = f"{path}/data/videos"
listdir = os.listdir(videos)

# parameter
interval = 2

count = 1
for rep in tqdm(listdir):
    cap = cv2.VideoCapture(videos+'/'+rep)

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if not count % interval:
                cv2.imwrite(f"{images}/{int(count/interval)}.jpg", frame)
            count += 1
        else:
            break

    cap.release()
