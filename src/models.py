import torch
from torch import nn
from torch.functional import F


class Discriminator(nn.Module):
  def __init__(self, image_size, hidden_size):
    super(Discriminator, self).__init__()
    first_lin_layer = nn.Linear(image_size, hidden_size)
    hidden_lin_layer = nn.Linear(hidden_size, hidden_size)
    last_lin_layer = nn.Linear(hidden_size, 1)
    self.lin_layers = nn.ModuleList(
        [first_lin_layer, hidden_lin_layer, last_lin_layer])

  def forward(self, x):
    for lin_layer in self.lin_layers[:-1]:
      x = lin_layer(x)
      x = F.leaky_relu(x, 0.2)
    x = self.lin_layers[-1](x)
    x = torch.sigmoid(x)
    return x


class Generator(nn.Module):
  def __init__(self, image_size, latent_size, hidden_size):
    super(Generator, self).__init__()
    first_lin_layer = nn.Linear(latent_size, hidden_size)
    hidden_lin_layer = nn.Linear(hidden_size, hidden_size)
    last_lin_layer = nn.Linear(hidden_size, image_size)
    self.lin_layers = nn.ModuleList(
        [first_lin_layer, hidden_lin_layer, last_lin_layer])

  def forward(self, x):
    for lin_layer in self.lin_layers[:-1]:
      x = lin_layer(x)
      x = F.relu(x)
    x = self.lin_layers[-1](x)
    x = torch.tanh(x)
    return x
