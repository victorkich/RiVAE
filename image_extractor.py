from tqdm import tqdm
import cv2
import os

local = os.getcwd()
images = local+'/data/images'
videos = local+'/data/videos'
listdir = os.listdir(videos)

count = 0
for rep in tqdm(listdir):
    cap = cv2.VideoCapture(videos+'/'+rep)

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(images+'/green/{}.jpg'.format(count), frame[0])
            cv2.imwrite(images+'/blue/{}.jpg'.format(count), frame[1])
            cv2.imwrite(images+'/red/{}.jpg'.format(count), frame[2])

            count += 1
        else:
            break

    cap.release()
