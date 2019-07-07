import os
import torch
import torchvision
import torch.nn as nn
import click

import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from src import models, utils

params = {
    'seed': 123456789,
    'batch_size': 100,
    'optimizer': 'adam',
    'lr': 2e-4,
    'wd': 0,
    'epochs': 200
}


@click.group()
def main():
  np.random.seed(params['seed'])
  torch.manual_seed(params['seed'])
  torch.cuda.manual_seed_all(params['seed'])
  torch.backends.cudnn.benchmark = True  # 学習最適化を行う．高速化．CNNでは推奨


@main.command()
def train():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  latent_size = 64  # ノイズ
  hidden_size = 256
  image_size = 784
  sample_dir = 'samples'

  os.makedirs(sample_dir, exist_ok=True)

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.5],
          std=[0.5])
  ])

  mnist = torchvision.datasets.MNIST(
      root='data',
      train=True,
      transform=transform,
      download=True
  )

  data_loader = torch.utils.data.DataLoader(
      dataset=mnist,
      batch_size=params['batch_size'],
      shuffle=True
  )

  D = models.Discriminator(image_size, hidden_size)
  G = models.Generator(image_size, latent_size, hidden_size)

  D = D.to(device)
  G = G.to(device)

  criterion = nn.BCELoss()
  d_optimizer = utils.get_optim(params, D)
  g_optimizer = utils.get_optim(params, G)

  total_step = len(data_loader)
  for epoch in range(params['epochs']):
    for i, (images, _) in enumerate(data_loader):  # labelは使わない
      # (batch_size, 1, 28, 28) -> (batch_size, 1*28*28)
      images = images.reshape(params['batch_size'], -1).to(device)

      real_labels = torch.ones(params['batch_size'], 1).to(device)
      fake_labels = torch.zeros(params['batch_size'], 1).to(device)

      # Train discriminator
      outputs = D(images)
      d_loss_real = criterion(outputs, real_labels)
      real_score = outputs

      z = torch.randn(params['batch_size'], latent_size).to(device)
      fake_images = G(z)
      outputs = D(fake_images)
      d_loss_fake = criterion(outputs, fake_labels)
      fake_score = outputs

      d_loss = d_loss_real + d_loss_fake
      d_optimizer.zero_grad()
      g_optimizer.zero_grad()
      d_loss.backward()
      d_optimizer.step()

      # Train generator
      z = torch.randn(params['batch_size'], latent_size).to(device)
      fake_images = G(z)
      outputs = D(fake_images)

      g_loss = criterion(outputs, real_labels)

      d_optimizer.zero_grad()
      g_optimizer.zero_grad()
      g_loss.backward()
      g_optimizer.step()

      if (i + 1) % 200 == 0:
        print('Rpoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
              .format(epoch, params['epochs'], i + 1, total_step, d_loss.item(), g_loss.item(),
                      real_score.mean().item(), fake_score.mean().item()))  # .item():ゼロ次元Tensorから値を取り出す

    if (epoch + 1) == 1:
      images = images.reshape(params['batch_size'], 1, 28, 28)
      save_image(utils.denorm(images), os.path.join(
          sample_dir, 'real_images.png'))
    fake_images = fake_images.reshape(params['batch_size'], 1, 28, 28)
    save_image(utils.denorm(fake_images), os.path.join(
        sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

  torch.save(G.state_dict(), 'G.ckpt')
  torch.save(D.state_dict(), 'D.ckpt')


if __name__ == '__main__':
  main()
