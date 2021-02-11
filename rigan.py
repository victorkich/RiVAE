from model import Generator, Discriminator
from dataset_creator import RiVAEDataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from os import path
import torch

# get image path
path_ = path.abspath(path.dirname(__file__))
img_path = f"{path_}/data/images"

# parameters
img_shape = (40, 3, 64, 64)  # (batch_size, c, w, h)
epochs = 100
lr = 0.0002
checkpoint_interval = 10

#  use gpu if available
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print("PyTorch CUDA:", cuda)
cudnn.benchmark = cuda


# custom weights initialization called on netG and netD
def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight, 0., 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1., 0.02)
        nn.init.zeros_(m.bias)


# create a model from RiGAN auto encoder class
netG = Generator(img_shape[2]).to(device)
netG.apply(weights_init)
netD = Discriminator(img_shape[2]).to(device)
netD.apply(weights_init)

# setup optimizer
optimizerD = Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# binary cross entropy loss
criterion = nn.BCELoss()

# define real and fake labels
real_label = 1
fake_label = 0

# loading the dataset using data loader
dataset = RiVAEDataset(img_dir=img_path, img_shape=img_shape)
dataloader = DataLoader(dataset, batch_size=img_shape[0], shuffle=True, num_workers=2)

for epoch in range(1, epochs+1):
    print(f"Epoch {epoch} of {epochs}")
    loss_generator = 0.
    loss_discriminator = 0.
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        # update discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        data = data.to(device)  # train with real
        data = data.reshape(img_shape)
        label = torch.full((img_shape[0],), real_label, dtype=data.dtype, device=device)
        output = netD(data)
        errD_real = criterion(output, label)
        errD_real.backward()

        # train with fake
        noise = torch.randn(img_shape[0], 100, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        loss_discriminator += errD
        optimizerD.step()

        # update generator network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        loss_generator += errG
        errG.backward()
        optimizerG.step()

        # save the last batch input and output of every epoch
        if i == len(dataloader) - 1:
            both = torch.cat((fake.detach(), data))
            save_image(both.cpu(), f"{path_}/data/outputs/output_{epoch}.png")

    print(f"Generator Loss: {loss_generator:.4f}\nDiscriminator Loss: {loss_discriminator:.4f}")
    if not epoch % checkpoint_interval:
        torch.save(netG.state_dict(), f"{path_}/models/generator_{epoch}.pth")
        torch.save(netD.state_dict(), f"{path_}/models/discriminator_{epoch}.pth")
