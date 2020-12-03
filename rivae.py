from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from dataset_creator import RiVAEDataset
from torch import optim
from torch import nn
from tqdm import tqdm
import pandas as pd
import torch
import model
import os

# get image path
path = os.path.abspath(os.path.dirname(__file__))
img_path = f"{path}/data/images"

# parameters
img_shape = [280, 280, 3]  # [h, w, c]
latent_dim = 1587
epochs = 1000
batch_size = 1
lr = 0.0001

#  use gpu if available
cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
print("PyTorch CUDA:", cuda_available)

# create a model from LinearVAE autoencoder class
# load it to the specified device, either gpu or cpu
model = model.RiVAE(latent_dim=latent_dim, batch_size=batch_size, img_shape=img_shape).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

# Binary Cross Entropy loss
criterion = nn.BCELoss(reduction='sum')

# loading the dataset using DataLoader
dataset = RiVAEDataset(img_dir=img_path, img_shape=img_shape)
lengths = [round(len(dataset)*0.8), round(len(dataset)*0.2)]
train_data, val_data = random_split(dataset, lengths, generator=torch.Generator())
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)


def kl_loss(mu, logvar):
    """
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
    """
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data = data.to(device)
        data = data.reshape((1, 3, data.shape[3], data.shape[2]))
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        criterion_loss = criterion(reconstruction, data)
        kl_div = kl_loss(mu, logvar)
        loss = criterion_loss + kl_div
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss


def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
            data = data.to(device)
            data = data.view((1, 3, data.shape[3], data.shape[2]))
            reconstruction, mu, logvar = model(data)
            criterion_loss = criterion(reconstruction, data)
            kl_div = kl_loss(mu, logvar)
            loss = criterion_loss + kl_div
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(val_data) / dataloader.batch_size) - 1:
                both = torch.cat((data.view(batch_size, img_shape[2], img_shape[0], img_shape[1]),
                                  reconstruction.view(batch_size, img_shape[2], img_shape[0], img_shape[1])))
                save_image(both.cpu(), f"{path}/data/outputs/output_{epoch}.png")
    val_loss = running_loss / len(dataloader.dataset)
    return val_loss


train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, train_loader)
    val_epoch_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    df_loss = pd.DataFrame({'train_loss': train_loss, 'val_loss': val_loss})
    df_loss.to_csv('output.csv')
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")
    torch.save(model.state_dict(), f"{path}/models/model_{epoch}.pth")
