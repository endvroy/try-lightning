import os
from argparse import Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import pytorch_lightning as pl


class Generator(nn.Module):
    def __init__(self, latent_dim, conv_channels, out_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, conv_channels[0],
                               4, stride=1, bias=False),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(True),
            nn.ConvTranspose2d(conv_channels[0], conv_channels[1],
                               4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels[1]),
            nn.ReLU(True),
            nn.ConvTranspose2d(conv_channels[1], conv_channels[2],
                               4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels[2]),
            nn.ReLU(True),
            nn.ConvTranspose2d(conv_channels[2], out_channels,
                               2, stride=2, padding=2, bias=False),
            nn.ReLU(True)
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, inp_channels, conv_channels):
        super(Discriminator, self).__init__()
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


class OldDiscriminator(nn.Module):
    def __init__(self,inp_channels, conv_channels):
        super(OldDiscriminator, self).__init__()
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
            nn.Sigmoid()
        )

    def forward(self, inp):
        return self.main(inp)


class GAN(pl.LightningModule):
    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        # networks
        self.generator = Generator(self.hparams.latent_dim, self.hparams.gen_conv_channels, 1)
        self.discriminator = OldDiscriminator(1, self.hparams.disc_conv_channels)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

        # self.example_input_array = torch.zeros((3, 100, 1, 1))

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_i):
        imgs, _ = batch
        self.last_imgs = imgs

        # train generator
        if optimizer_i == 0:
            # sample noise
            z = torch.randn(imgs.size(0), self.hparams.latent_dim, 1, 1).to(imgs.device)

            # generate images
            self.generated_imgs = self.forward(z)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1).to(imgs.device)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_i == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(imgs.device.index)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1).to(imgs.device)

            fake_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs.detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    @pl.data_loader
    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    def on_epoch_end(self):
        z = torch.randn(32, self.hparams.latent_dim, 1, 1).to(self.last_imgs.device)

        # log sampled images
        sample_imgs = self.forward(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)


if __name__ == '__main__':
    args = {
        'batch_size': 64,
        'lr': 1e-2,
        'latent_dim': 100,
        'gen_conv_channels': [80, 40, 20],
        'disc_conv_channels': [16, 32, 16]
    }
    hparams = Namespace(**args)
    gan_model = GAN(hparams)

    from pytorch_lightning.callbacks import ModelCheckpoint

    save_path = 'old_dcgan_logs_v2'
    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=f'{save_path}/checkpoints',
        save_best_only=False,
        verbose=True,
    )

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(gpus=1,
                         default_save_path=f'{save_path}',
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=None,
                         max_nb_epochs=20)
    # gan_model.summary()
    trainer.fit(gan_model)
