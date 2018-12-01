import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.ngf = 64
        self.netG = nn.Sequential(
                    nn.ConvTranspose2d(self.noise_dim, self.ngf * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(self.ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.ngf * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.ngf),
                    nn.ReLU(True),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (num_channels) x 64 x 64
                    )
        def forward(self, z):
            return self.netG(z)