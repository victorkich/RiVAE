from cv2 import VideoCapture, VideoWriter, VideoWriter_fourcc, hconcat
from torch import load, cuda, device
from argparse import ArgumentParser
from numpy import zeros
from tqdm import tqdm
import model
import os

path = os.path.abspath(os.path.dirname(__file__))

# Construct the argument parser
ap = ArgumentParser()

# Add the arguments to the parser
ap.add_argument("--video", required=True, help="video for test the model")
ap.add_argument("--model", required=True, help="model used in the test")
ap.add_argument("--latent_dim", required=True, help="latent dimension of the model")
args = vars(ap.parse_args())

path_video = f"{path}/data/videos/{args['video']}"
path_model = f"{path}/models/{args['model']}"

# Create a VideoCapture object
cap = VideoCapture(path_video)

# Use gpu if available
cuda_available = cuda.is_available()
device = device('cuda' if cuda_available else 'cpu')
print("PyTorch CUDA:", cuda_available)

# Load model
model = model.RiVAE(latent_dim=int(args['latent_dim'])).to(device)
model.load_state_dict(load(path_model))
model.eval()

# Define the codec and create VideoWriter object.The output is stored in 'test_model.mp4' file.
out = VideoWriter('test_model.mp4', VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (cap.get(3) >> 1, cap.get(4)))

border = zeros((10, cap.get(4), 3))

for _ in tqdm(range(cap.get(7))):
    ret, frame = cap.read()
    if ret:
        reconstruction = model(frame)
        final_frame = hconcat([frame, border, reconstruction])
        out.write(final_frame)
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()
