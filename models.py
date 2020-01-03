import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SimpleGenerator(nn.Module):
    def block(self, in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def __init__(self, latent_dim):
        super(SimpleGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            *self.block(latent_dim, 128, normalize=False),
            *self.block(128, 256),
            *self.block(256, 512),
            *self.block(512, 1024),
            nn.Linear(1024, 28 * 28),
            nn.Sigmoid()
        )

    def sample_z(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim)
        return z

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class SimpleDiscriminator(nn.Module):
    def __init__(self, sigm=False):
        super(SimpleDiscriminator, self).__init__()

        self.sigm = sigm
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)

        if self.sigm:
            validity = torch.sigmoid(validity)
        return validity


class RepeatRGB(nn.Module):
    def forward(self, x):
        return x.expand((-1, 3, -1, -1))


class ToyGen(nn.Module):
    def __init__(self, latent_dim):
        super(ToyGen, self).__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 4),  # 4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 2, 1),  # 7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),  # 28
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

    def sample_z(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim, 1, 1)
        return z


class ToyDisc(nn.Module):
    def __init__(self, sigm=False):
        super(ToyDisc, self).__init__()
        self.sigm = sigm
        linear = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 11),
        )

        vgg = torchvision.models.vgg11(pretrained=False)
        conv = vgg.features[:6]

        self.model = nn.Sequential(
            RepeatRGB(),
            conv,
            nn.Flatten(),
            linear,
        )

    def forward(self, x):
        x = self.model(x)
        if self.sigm:
            x = torch.sigmoid(x)
        return x


class MyDCGen(nn.Module):
    def __init__(self, latent_dim, conv_channels, out_channels):
        super(MyDCGen, self).__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, conv_channels[0],
                               4, stride=1, bias=False),
            nn.BatchNorm2d(conv_channels[0]),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(conv_channels[0], conv_channels[1],
                               4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels[1]),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(conv_channels[1], conv_channels[2],
                               4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels[2]),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(conv_channels[2], out_channels,
                               2, stride=2, padding=2, bias=False),
            nn.Sigmoid()
        )

    def sample_z(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim, 1, 1)
        return z

    def forward(self, z):
        img = self.model(z)
        return img


class MyDCDisc(nn.Module):
    def __init__(self, inp_channels, conv_channels):
        super(MyDCDisc, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(inp_channels, conv_channels[0], (2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(conv_channels[0], conv_channels[1], (2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(conv_channels[1]),
            nn.Conv2d(conv_channels[1], conv_channels[2], (5, 5), dilation=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(conv_channels[2], conv_channels[3], (4, 4), stride=4),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(conv_channels[3]),
            nn.Conv2d(conv_channels[3], conv_channels[4], (2, 2), dilation=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(conv_channels[4], 1, (2, 2)),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        return out


class MyOldDCDisc(nn.Module):
    def __init__(self, inp_channels, conv_channels, sigm=False):
        super(MyOldDCDisc, self).__init__()
        self.sigm = sigm
        self.main = nn.Sequential(
            # input is (gan_args.nc) x 28 x 28
            nn.Conv2d(inp_channels, conv_channels[0], 4, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. gan_args.ndf x 16 x 16
            nn.Conv2d(conv_channels[0], conv_channels[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_channels[1]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (gan_args.ndf*2) x 8 x 8
            nn.Conv2d(conv_channels[1], conv_channels[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_channels[2]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (gan_args.ndf*4) x 4 x 4
            nn.Conv2d(conv_channels[2], 1, 4, 1, 0, bias=False),
        )

    def forward(self, inp):
        x = self.main(inp)
        if self.sigm:
            x = torch.sigmoid(x)
        return x


class DCGen(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_channels):
        super(DCGen, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, hidden_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(hidden_dim, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def sample_z(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim, 1, 1)
        return z

    def forward(self, z):
        x = self.main(z)
        # x = F.interpolate(x, size=(28, 28), mode='bilinear')
        return x


class DCDisc(nn.Module):
    def __init__(self, hidden_dim, n_channels, sigm=False):
        super(DCDisc, self).__init__()
        self.sigm = sigm
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(n_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        x = self.main(x)
        if self.sigm:
            x = torch.sigmoid(x)
        return x
