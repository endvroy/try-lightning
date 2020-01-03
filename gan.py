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
from models import DCGen as Generator, DCDisc as Discriminator


class GAN(pl.LightningModule):
    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        # networks
        self.generator = Generator(self.hparams.latent_dim, self.hparams.model_dim, 1)
        self.discriminator = Discriminator(self.hparams.model_dim, 1, sigm=True)

        self.critic_counter = 0

        # self.example_input_array = torch.zeros((3, 100, 1, 1))

    def sample_z(self, batch_size):
        z = self.generator.sample_z(batch_size).to(next(self.generator.parameters()).device)
        return z

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, disc_responses):
        return F.binary_cross_entropy(disc_responses, torch.ones_like(disc_responses))

    def minmax_loss(self, real_scores, fake_scores):
        return F.binary_cross_entropy(real_scores, torch.ones_like(real_scores)) + \
               F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores))

    def grad_penalty(self, generated_imgs, real_imgs):
        alpha = torch.rand(real_imgs.shape[0], 1, 1, 1).to(real_imgs.device)
        interpolated = alpha * real_imgs + (1 - alpha) * generated_imgs
        interpolated.requires_grad_(True)
        disc_interpolates = self.discriminator(interpolated)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolated,
                                        grad_outputs=torch.ones_like(disc_interpolates).to(real_imgs.device),
                                        create_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean()
        return grad_penalty

    def training_step(self, batch, batch_nb):
        imgs, _ = batch

        # train generator
        if self.optim_idx == 0:
            # sample noise
            z = self.sample_z(imgs.size(0))
            # generate images
            generated_imgs = self.forward(z)
            # adversarial loss
            critic_responses = self.discriminator(generated_imgs)
            g_loss = self.adversarial_loss(critic_responses)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if self.optim_idx == 1:
            with torch.no_grad():
                z = self.sample_z(imgs.size(0))
                generated_imgs = self.forward(z)

            # Measure discriminator's ability to classify real from generated samples
            real_scores = self.discriminator(imgs)

            # how well can it label as fake?
            fake_scores = self.discriminator(generated_imgs)

            minmax_loss = self.minmax_loss(real_scores, fake_scores)
            # grad_penalty = self.grad_penalty(generated_imgs, imgs)
            d_loss = minmax_loss

            tqdm_dict = {'d_loss': d_loss,
                         # 'gp': grad_penalty
                         }
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
        transform = transforms.Compose([transforms.Resize(64),
                                        transforms.ToTensor(),
                                        # transforms.Normalize([0.5], [0.5])
                                        ])
        dataset = MNIST('.', train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)

    def on_batch_start(self, batch):
        self.optimizers_bak = self.trainer.optimizers
        self.optim_idx = self.trainer.batch_nb % 2
        self.trainer.optimizers = [self.optimizers_bak[self.optim_idx]]
        self.trainer.optimizers[0].zero_grad()
        pass

    def on_batch_end(self):
        self.trainer.optimizers = self.optimizers_bak

    def on_epoch_end(self):
        z = self.sample_z(32)

        # log sampled images
        with torch.no_grad():
            sample_imgs = self.forward(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)


if __name__ == '__main__':
    args = {
        'batch_size': 128,
        'lr': 0.0002,
        'latent_dim': 100,
        'model_dim': 64,
        'gp_lambda': 10,
        'disc_per_gen': 1
    }
    hparams = Namespace(**args)
    gan_model = GAN(hparams)

    from pytorch_lightning.callbacks import ModelCheckpoint

    save_path = 'gan_dc_logs'
    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=f'logs/{save_path}/checkpoints',
        save_best_only=False,
        verbose=True,
    )

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(gpus=1,
                         default_save_path=f'logs/{save_path}',
                         early_stop_callback=False,
                         max_nb_epochs=30,
                         row_log_interval=1,
                         # print_nan_grads=True
                         )
    # gan_model.summary()
    trainer.fit(gan_model)
