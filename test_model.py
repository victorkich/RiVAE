from torch import load, cuda, device
from argparse import ArgumentParser
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import model
import cv2
import os

# Parameters
img_shape = (1, 3, 512, 512)

path = os.path.abspath(os.path.dirname(__file__))

# Construct the argument parser
ap = ArgumentParser()

# Add the arguments to the parser
ap.add_argument("--video", required=True, help="video for test the model")
ap.add_argument("--model", required=True, help="model used in the test")
ap.add_argument("--output", required=True, help="name for output video")
args = vars(ap.parse_args())

path_video = f"{path}/data/videos/{args['video']}"
path_model = f"{path}/models/{args['model']}"

# Create a VideoCapture object
cap = cv2.VideoCapture(path_video)

# Use gpu if available
cuda_available = cuda.is_available()
device = device('cuda' if cuda_available else 'cpu')
print("PyTorch CUDA:", cuda_available)

# Load model
model = model.RiVAE().to(device)
model.load_state_dict(load(path_model))
model.eval()

# Define the codec and create VideoWriter object.The output is stored in 'test_model.mp4' file.
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(args['output'], fourcc, 30.0, (img_shape[3]*2, img_shape[2]))

trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_shape[2], img_shape[3]))
])

for _ in tqdm(range(int(cap.get(7)))):
    ret, frame = cap.read()
    if ret:
        data = trans(Image.fromarray(frame.astype(np.uint8)))
        data = data.to(device)
        data = data.view(img_shape)
        reconstruction, _, _ = model(data)
        reconstruction = np.uint8(reconstruction.cpu().detach().numpy() * 255).squeeze()
        reconstruction = cv2.merge((reconstruction[2, :, :], reconstruction[1, :, :], reconstruction[0, :, :]))
        frame = cv2.resize(frame, (img_shape[2], img_shape[3]))
        final_frame = np.hstack([frame, reconstruction])
        cv2.imshow("Frame", final_frame)
        out.write(final_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()
