from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from dataset_creator import RiVAEDataset
from model import RiVAE
from torch import optim
from tqdm import tqdm
from torch import nn
import pandas as pd
from os import path
import torch

# get image path
path_ = path.abspath(path.dirname(__file__))
img_path = f"{path_}/data/images"

# parameters
img_shape = (40, 3, 64, 64)  # (batch_size, c, w, h)
epochs = 100
lr = 0.0004
checkpoint_interval = 10

#  use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("PyTorch CUDA:", torch.cuda.is_available())

# create a model from RiVAE auto encoder class
model = RiVAE().to(device)

# Adam optimizer with learning rate 4e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

# Mean Squared Error loss
criterion = nn.MSELoss(reduction='sum')

# loading the dataset using DataLoader
dataset = RiVAEDataset(img_dir=img_path, img_shape=img_shape)
lengths = [round(len(dataset)*0.8), round(len(dataset)*0.2)]
train_data, val_data = random_split(dataset, lengths, generator=torch.Generator())
train_loader = DataLoader(train_data, batch_size=img_shape[0], shuffle=True)
val_loader = DataLoader(val_data, batch_size=img_shape[0], shuffle=True)


def kl_loss(mu, log_var):
    """
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        mu: the mean from the latent vector
        log_var: log variance from the latent vector
    """
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl_div


def fit(net, data_l):
    net.train()
    running_loss = 0.0
    for data in tqdm(data_l, total=int(len(train_data)/data_l.batch_size)):
        data = data.to(device)
        data = data.reshape(img_shape)
        optimizer.zero_grad()
        reconstruction, mu, log_var = net(data)
        criterion_loss = criterion(reconstruction, data)
        kl_div = kl_loss(mu, log_var)
        loss = criterion_loss + kl_div
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    t_loss = running_loss / len(data_l.dataset)
    return t_loss


def validate(net, data_l):
    net.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_l), total=int(len(val_data)/data_l.batch_size)):
            data = data.to(device)
            data = data.view(img_shape)
            reconstruction, mu, log_var = net(data)
            criterion_loss = criterion(reconstruction, data)
            kl_div = kl_loss(mu, log_var)
            loss = criterion_loss + kl_div
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(val_data)/data_l.batch_size) - 1:
                both = torch.cat((data.view(img_shape), reconstruction.view(img_shape)))
                save_image(both.cpu(), f"{path}/data/outputs/output_{epoch}.png")
    v_loss = running_loss / len(data_l.dataset)
    return v_loss


train_loss = []
val_loss = []
for epoch in range(1, epochs+1):
    print(f"Epoch {epoch} of {epochs}")
    train_epoch_loss = fit(model, train_loader)
    val_epoch_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    df_loss = pd.DataFrame({'train_loss': train_loss, 'val_loss': val_loss})
    df_loss.to_csv('output.csv')
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")
    if not epoch % checkpoint_interval:
        torch.save(model.state_dict(), f"{path}/models/model_{epoch}.pth")
