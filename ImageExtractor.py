import cv2
import os

local = os.getcwd()
images = local+'/data/images'
videos = local+'/data/videos'
listdir = os.listdir(videos)

count = 0
for rep in listdir:
    cap = cv2.VideoCapture(videos+'/'+rep+'.mp4')

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(images+'/{}.jpg'.format(count), frame)
            count += 1
        else:
            break

    cap.release()
