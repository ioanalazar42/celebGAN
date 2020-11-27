'''Architecture of GAN Discriminator and Generator'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    '''Input to discriminator is a 3x128x128 image and returns scalar between [0, 1]
    representing the probability if the image is fake (generated) > 0 or real > 1'''

    def __init__(self):
        super(Discriminator, self).__init__()

        # input is 3x128x128, output is 64x32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(6, 6), stride=4, padding=1)

        # input is 64x32x32, output is 128x16x16
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)

        # input is 128x16x16, output is 256x8x8
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(256)

        # input is 256x8x8, output is 512x4x4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(512)

        # input is 512x4x4, output is 1x1x1
        self.conv5 = nn.Conv2d(512, 1, kernel_size=(4, 4), stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2_bn(F.relu(self.conv2(x)))
        x = self.conv3_bn(F.relu(self.conv3(x)))
        x = self.conv4_bn(F.relu(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))

        # reinterpret Nx1x1 as N
        return x.squeeze()

class Generator(nn.Module):
    ''''Takes 512-dimensional latent space vector and generates a 3x128x128 image.'''
    def __init__(self):
        super(Generator, self).__init__()

        # input is a latent space vector of size 512, output is 512x4x4
        self.convtranspose1 = nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=1)
        self.convtranspose1_bn = nn.BatchNorm2d(512)

        # input is 512x4x4, output is 256x8x8
        self.convtranspose2 = nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=2, padding=1)
        self.convtranspose2_bn = nn.BatchNorm2d(256)

        # input is 256x8x8, output is 128x16x16
        self.convtranspose3 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=2, padding=1)
        self.convtranspose3_bn = nn.BatchNorm2d(128)

        # input is 128x16x16, output 64x32x32
        self.convtranspose4 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1)
        self.convtranspose4_bn = nn.BatchNorm2d(64)

        # input is 64x32x32, output 3x128x128
        self.convtranspose5 = nn.ConvTranspose2d(64, 3, kernel_size=(6, 6), stride=4, padding=1)

    def forward(self, x):
        x = self.convtranspose1_bn(F.relu(self.convtranspose1(x)))
        x = self.convtranspose2_bn(F.relu(self.convtranspose2(x)))
        x = self.convtranspose3_bn(F.relu(self.convtranspose3(x)))
        x = self.convtranspose4_bn(F.relu(self.convtranspose4(x)))
        x = torch.tanh(self.convtranspose5(x))

        return x
