import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(SimpleGenerator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class SimpleDiscriminator(nn.Module):
    def __init__(self, img_shape):
        super(SimpleDiscriminator, self).__init__()

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

        return validity


class RepeatRGB(nn.Module):
    def forward(self, x):
        return x.expand((-1, 3, -1, -1))


class ToyGen(nn.Module):
    def __init__(self, latent_dim):
        super(ToyGen, self).__init__()
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


class ToyDisc(nn.Module):
    def __init__(self):
        super(ToyDisc, self).__init__()
        linear = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 11),
        )

        vgg = torchvision.models.vgg11(pretrained=False)
        # conv = vgg.features[:11]
        conv = vgg.features[:6]

        self.model = nn.Sequential(
            RepeatRGB(),
            conv,
            nn.Flatten(),
            linear,
        )

    def forward(self, x):
        return self.model(x)
