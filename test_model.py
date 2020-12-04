from tqdm import tqdm
import argparse
import torch
import model
import cv2
import os

path = os.path.abspath(os.path.dirname(__file__))

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("--video", required=True, help="video for test the model")
ap.add_argument("--model", required=True, help="model used in the test")
args = vars(ap.parse_args())

path_video = f"{path}/data/videos/{args['video']}"
path_model = f"{path}/models/{args['model']}"

# Create a VideoCapture object
cap = cv2.VideoCapture(path_video)

# Use gpu if available
cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
print("PyTorch CUDA:", cuda_available)

# Load model
model = model.RiVAE(latent_dim=1587).to(device)
model.load_state_dict(torch.load(path_model))
model.eval()

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'test_model.mp4' file.
out = cv2.VideoWriter('test_model.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width >> 1, frame_height))

for _ in tqdm(range(cap.get(7))):
    ret, frame = cap.read()
    if ret:
        reconstruction = model(frame)
        final_frame = cv2.hconcat([frame, reconstruction])
        out.write(final_frame)
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()
