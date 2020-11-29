import torch.nn.functional as F
import torch.nn as nn
import torch


class RiVAE(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(RiVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        linear_size = (img_shape[0]-4)*(img_shape[1]-4)*img_shape[2]

        # encoder
        self.enc1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)  # 18x10x3
        self.enc2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)  # 16x8x3
        self.enc3 = nn.Flatten()
        self.enc4 = nn.Linear(in_features=linear_size, out_features=latent_dim << 1)

        # decoder
        self.dec1 = nn.Linear(in_features=latent_dim, out_features=linear_size)  # 16x8x3
        self.dec2 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3)   # 18x10x3
        self.dec3 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.enc3(x)
        x = self.enc4(x).view(-1, 2, self.latent_dim)
        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.dec1(z))
        x = x.reshape((1, self.img_shape[2], self.img_shape[0]-4, self.img_shape[1]-4))
        x = F.relu(self.dec2(x))
        reconstruction = self.dec3(x)
        return reconstruction, mu, log_var
