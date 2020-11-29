from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset_creator import RivaeDataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
import model
import os

local = os.getcwd()
img_path = f"{local}/data/images"

# parameters
epochs = 10
batch_size = 1
workers = 1
lr = 0.0001

#  use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create a model from LinearVAE autoencoder class
# load it to the specified device, either gpu or cpu
model = model.LinearVAE().to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

# Binary Cross Entropy loss
criterion = nn.BCELoss(reduction='sum')

dataset = RivaeDataset(img_path)
lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
train_data, val_data = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator())

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=workers)


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
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
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(val_data) / dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 1, 28, 28)[:8],
                                  reconstruction.view(batch_size, 1, 28, 28)[:8]))
                save_image(both.cpu(), f"../outputs/output{epoch}.png", nrow=num_rows)
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
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")
