import torch.nn as nn
import torch


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class PrintShape(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


class RiVAE(nn.Module):
    def __init__(self):
        super(RiVAE, self).__init__()

        # Output = ((I - K + 2P) / S + 1)
        # I - a size of input neuron
        # K - kernel size
        # P - padding
        # S - stride

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Flatten()
        )

        # mu and log_var
        self.mu = nn.Linear(in_features=16384, out_features=50)
        self.log_var = nn.Linear(in_features=16384, out_features=50)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=50, out_features=16384),
            View((-1, 64, 16, 16)),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def reparameterize(self, mu, log_var):
        """
            mu: mean from the encoder's latent space
            log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # randn_like as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        x = self.encoder(x)

        # get mu and log_var
        mu = self.mu(x)
        log_var = self.log_var(x)

        # get the latent vector through re-parameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var


class Generator(nn.Module):
    def __init__(self, shape):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=shape >> 3, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(num_features=shape >> 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=shape >> 3, out_channels=shape >> 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=shape >> 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=shape >> 2, out_channels=shape >> 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=shape >> 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=shape >> 1, out_channels=shape, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=shape),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=shape, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.network(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, shape):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=shape, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=shape, out_channels=shape >> 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=shape >> 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=shape >> 1, out_channels=shape >> 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=shape >> 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=shape >> 2, out_channels=shape >> 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=shape >> 3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=shape >> 3, out_channels=1, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)
