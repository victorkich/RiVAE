import torch.nn as nn
import torch


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class PrintShape(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


class RiVAE(nn.Module):
    def __init__(self, latent_dim, batch_size, img_shape):
        super(RiVAE, self).__init__()

        '''
        Output = ((I - K + 2P) / S + 1)
        I - a size of input neuron
        K - kernel size
        P - padding
        S - stride
        '''

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(24, 12, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=1),
            nn.Flatten(),
            nn.Linear(in_features=6348, out_features=latent_dim << 1),
            nn.Sigmoid(),
            View((-1, 2, latent_dim))
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=6348),
            nn.ReLU(inplace=True),
            View((batch_size, 12, 23, 23)),
            nn.ConvTranspose2d(12, 24, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 12, kernel_size=5, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(12, 3, kernel_size=2, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # randn_like as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        x = self.encoder(x)

        # get `mu` and `log_var`
        mu = x[:, 0, :]
        log_var = x[:, 1, :]

        # get the latent vector through re-parameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var
