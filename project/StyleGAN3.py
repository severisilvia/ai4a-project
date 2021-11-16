import torch
from torch import nn


class StyleGAN3_Generator(nn.Module):
    def __init__(self):
        super(StyleGAN3_Generator, self).__init__()

        # IMPEMENTARE QUI IL GENERATORE DI STYLEGAN3
        self.conv = nn.Sequential()

    def forward(self, x):
        return self.conv(x)


class StyleGAN3_Discriminator(nn.Module):
    def __init__(self):
        super(StyleGAN3_Discriminator, self).__init__()

        # IMPLEMENTARE QUI IL DISCRIMINATORE DI STYLEGAN3
        self.main = nn.Sequential()

    def forward(self, input):
        return self.main(input)


# queste due sono da sistemare
def LSGAN_D(real, fake):
    return (torch.mean((real - 1) ** 2) + torch.mean(fake ** 2))


def LSGAN_G(fake):
    return torch.mean((fake - 1) ** 2)

def GANLoss():
    #loss di stylegan3
    return 0
