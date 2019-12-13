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
    def __init__(self, latent_dim, model_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim

        preprocess = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 4 * self.model_dim),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.model_dim, 2 * self.model_dim, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.model_dim, self.model_dim, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(self.model_dim, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        output = self.preprocess(z)
        output = output.view(-1, 4 * self.model_dim, 4, 4)
        # print output.size()
        output = self.block1(output)
        # print output.size()
        output = output[:, :, :7, :7]
        # print output.size()
        output = self.block2(output)
        # print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        # print output.size()
        return output


class Discriminator(nn.Module):
    def __init__(self, model_dim):
        super(Discriminator, self).__init__()
        self.model_dim = model_dim

        main = nn.Sequential(
            nn.Conv2d(1, self.model_dim, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*self.model_dim),
            nn.ReLU(True),
            nn.Conv2d(self.model_dim, 2 * self.model_dim, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*self.model_dim, 4*4*4*self.model_dim),
            nn.ReLU(True),
            nn.Conv2d(2 * self.model_dim, 4 * self.model_dim, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*self.model_dim, 4*4*4*self.model_dim),
            nn.ReLU(True),
            # nn.Linear(4*4*4*self.model_dim, 4*4*4*self.model_dim),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*self.model_dim, 4*4*4*self.model_dim),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4 * 4 * 4 * self.model_dim, 1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        out = self.main(x)
        out = out.view(-1, 4 * 4 * 4 * self.model_dim)
        out = self.output(out)
        return out.view(-1)


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
        'gp_lambda': 10
    }
    hparams = Namespace(**args)
    gan_model = GAN(hparams)

    from pytorch_lightning.callbacks import ModelCheckpoint

    save_path = 'wsgan_gp_logs_v2'
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
