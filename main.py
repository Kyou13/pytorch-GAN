import os
import torch
import torchvision
import torch.nn as nn
import click
import datetime
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
    'epochs': 200,
    'image_size': 784,
    'latent_size': 64,
    'hidden_size': 256
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

  D = models.Discriminator(params['image_size'], params['hidden_size'])
  G = models.Generator(params['image_size'],
                       params['latent_size'], params['hidden_size'])

  D = D.to(device)
  G = G.to(device)

  criterion = nn.BCELoss()
  d_optimizer = utils.get_optim(params, D)
  g_optimizer = utils.get_optim(params, G)

  total_step = len(data_loader)
  for epoch in range(params['epochs']):
    for i, (images, _) in enumerate(data_loader):  # labelは使わない
      # (batch_size, 1, 28, 28) -> (batch_size, 1*28*28)
      b_size = images.size(0)
      images = images.reshape(b_size, -1).to(device)

      real_labels = torch.ones(b_size, 1).to(device)
      fake_labels = torch.zeros(b_size, 1).to(device)

      # Train discriminator
      outputs = D(images)
      d_loss_real = criterion(outputs, real_labels)
      real_score = outputs

      z = torch.randn(b_size, params['latent_size']).to(device)
      fake_images = G(z.detach())
      outputs = D(fake_images)
      d_loss_fake = criterion(outputs, fake_labels)
      fake_score = outputs

      d_loss = d_loss_real + d_loss_fake
      d_optimizer.zero_grad()
      d_loss.backward()
      d_optimizer.step()

      # Train generator
      fake_images = G(z)
      outputs = D(fake_images)

      g_loss = criterion(outputs, real_labels)

      g_optimizer.zero_grad()
      g_loss.backward()
      g_optimizer.step()

      if (i + 1) % 200 == 0:
        print('Rpoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
              .format(epoch, params['epochs'], i + 1, total_step, d_loss.item(), g_loss.item(),
                      real_score.mean().item(), fake_score.mean().item()))  # .item():ゼロ次元Tensorから値を取り出す

    if (epoch + 1) == 1:
      images = images.reshape(b_size, 1, 28, 28)
      save_image(utils.denorm(images), os.path.join(
          sample_dir, 'real_images.png'))
    fake_images = fake_images.reshape(b_size, 1, 28, 28)
    save_image(utils.denorm(fake_images), os.path.join(
        sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

  torch.save(G.state_dict(), 'G.ckpt')
  torch.save(D.state_dict(), 'D.ckpt')


@main.command()
def generate():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  sample_dir = 'samples'
  os.makedirs(sample_dir, exist_ok=True)

  G = models.Generator(params['image_size'],
                       params['latent_size'], params['hidden_size'])
  G.load_state_dict(torch.load('G.ckpt'))
  G.eval()
  G = G.to(device)

  with torch.no_grad():
    z = torch.randn(params['batch_size'], params['latent_size']).to(device)
    fake_images = G(z)

  fake_images = fake_images.reshape(params['batch_size'], 1, 28, 28)
  dt_now = datetime.datetime.now()
  now_str = dt_now.strftime('%y%m%d%H%M%S')
  save_image(utils.denorm(fake_images), os.path.join(
      sample_dir, 'fake_images_{}.png'.format(now_str)))
  print('Saved Image ' + os.path.join(sample_dir,
                                      'fake_images_{}.png'.format(now_str)))


if __name__ == '__main__':
  main()
