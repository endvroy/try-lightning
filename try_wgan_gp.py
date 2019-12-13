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
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

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
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

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


class GAN(pl.LightningModule):
    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        # networks
        self.generator = Generator(self.hparams.latent_dim, self.hparams.model_dim)
        self.discriminator = Discriminator(self.hparams.model_dim)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

        self.critic_counter = 0

        # self.example_input_array = torch.zeros((3, 100, 1, 1))

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, critic_responses):
        return -torch.mean(critic_responses)

    def wasserstein_loss(self, real_scores, fake_scores):
        return torch.mean(fake_scores) - torch.mean(real_scores)

    def grad_penalty(self, imgs):
        alpha = torch.rand(imgs.shape[0], 1, 1, 1).to(imgs.device)
        interpolated = alpha * imgs + (1 - alpha) * self.generated_imgs.detach()
        interpolated.requires_grad = True
        disc_interpolates = self.discriminator(interpolated)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolated,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(imgs.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def training_step(self, batch, batch_nb, optimizer_i):
        imgs, _ = batch
        self.last_imgs = imgs

        # train generator
        if optimizer_i == 0:
            if self.critic_counter < hparams.disc_per_gen:
                self.critic_counter += 1
                return {'loss': torch.tensor(-1)}
            else:
                self.critic_counter = 0

            # sample noise
            z = torch.randn(imgs.size(0), self.hparams.latent_dim).to(imgs.device)

            # generate images
            self.generated_imgs = self.forward(z)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs))
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_i == 1:
            with torch.no_grad():
                z = torch.randn(imgs.size(0), self.hparams.latent_dim).to(imgs.device)
                self.generated_imgs = self.forward(z)

            # Measure discriminator's ability to classify real from generated samples
            real_scores = self.discriminator(imgs)

            # how well can it label as fake?
            fake_scores = self.discriminator(self.generated_imgs.detach())

            ws_loss = self.wasserstein_loss(real_scores, fake_scores)
            grad_penalty = self.grad_penalty(imgs)
            d_loss = ws_loss + grad_penalty

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def backward(self, use_amp, loss, optimizer):
        if loss == -1:
            return
        else:
            return super(GAN, self).backward(use_amp, loss, optimizer)

    def configure_optimizers(self):
        lr = self.hparams.lr
        beta_1 = self.hparams.beta_1
        beta_2 = self.hparams.beta_2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta_1, beta_2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))
        return [opt_g, opt_d], []

    @pl.data_loader
    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    def on_epoch_end(self):
        z = torch.randn(32, self.hparams.latent_dim).to(self.last_imgs.device)

        # log sampled images
        sample_imgs = self.forward(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)


if __name__ == '__main__':
    args = {
        'batch_size': 64,
        'lr': 1e-4,
        'beta_1': 0.5,
        'beta_2': 0.9,
        'latent_dim': 100,
        'model_dim': 128,
        'gp_lambda': 10,
        'disc_per_gen': 8
    }
    hparams = Namespace(**args)
    gan_model = GAN(hparams)

    from pytorch_lightning.callbacks import ModelCheckpoint

    save_path = 'wsgan_gp_logs_v5'
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
                         max_nb_epochs=100)
    # gan_model.summary()
    trainer.fit(gan_model)
