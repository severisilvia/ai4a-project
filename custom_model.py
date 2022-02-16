import torch
from utils.utils import weights_init_normal
from StyleGAN2 import Discriminator
from StyleGAN3 import Generator
import argparse
import itertools
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

from StyleGAN2 import Discriminator
from StyleGAN3 import Generator
from dataset import ImageDataset
# Per costruire i dizionari da mandare come argomenti del training
from utils.utils import EasyDict
from utils.utils import LambdaLR
# cerca di capire queste funzioni a cosa servono
from utils.utils import ReplayBuffer
from utils.utils import weights_init_normal
from utils.utils import tensor2image
from utils.utils import Logger

class custom_model(torch.nn.Module):

    def __init__(self, batchSize):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', help='Base configuration, possible choices: stylegan3-t, stylegan3-r,stylegan2',type=str, default='stylegan3-t')
        parser.add_argument('--cbase', help='Capacity multiplier', type=int, default=32768)
        parser.add_argument('--cmax', help='Max. feature maps', type=int, default=512)
        parser.add_argument('--map-depth', help='Mapping network depth  [default: varies]', type=int, default=2)
        parser.add_argument('--freezed', help='Freeze first layers of D', type=int, default=0)
        parser.add_argument('--mbstd-group', help='Minibatch std group size', type=int, default=4)
        parser.add_argument('--label_dim', help='Number of labels', type=int, default=0)
        parser.add_argument('--resolution',help='Resolution of the images expressed as the dimension of one of the two equals dimension image.shape[1] or image.shape[2] of the image, note that we want squared images obviously',
                            type=int, default=512)
        parser.add_argument('--num_channels', help='Number of channels of the data, so the image.shape[0]', type=int, default=3)
        opt = parser.parse_args()

        G_kwargs = EasyDict(z_dim=1000, w_dim=512, mapping_kwargs=EasyDict())
        D_kwargs = EasyDict(block_kwargs=EasyDict(), mapping_kwargs=EasyDict(), epilogue_kwargs=EasyDict())
        # Hyperparameters & settings.
        batch_size = batchSize
        G_kwargs.channel_base = D_kwargs.channel_base = opt.cbase
        G_kwargs.channel_max = D_kwargs.channel_max = opt.cmax
        G_kwargs.mapping_kwargs.num_layers = 2 if opt.map_depth is None else opt.map_depth
        D_kwargs.block_kwargs.freeze_layers = opt.freezed
        D_kwargs.epilogue_kwargs.mbstd_group_size = opt.mbstd_group
        # metrics = opts.metrics
        # Base configuration.
        G_kwargs.magnitude_ema_beta = 0.5 ** (batch_size / (20 * 1e3))
        if opt.cfg == 'stylegan3-r':
            G_kwargs.conv_kernel = 1  # Use 1x1 convolutions.
            G_kwargs.channel_base *= 2  # Double the number of feature maps.
            G_kwargs.channel_max *= 2
            G_kwargs.use_radial_filters = True  # Use radially symmetric downsampling filters.
        common_kwargs = dict(c_dim=opt.label_dim, img_resolution=opt.resolution, img_channels=opt.num_channels)

        self.netG_A2B = Generator(**G_kwargs, **common_kwargs)
        self.netG_B2A = Generator(**G_kwargs, **common_kwargs)
        self.netD_A = Discriminator(**D_kwargs, **common_kwargs)
        self.netD_B = Discriminator(**D_kwargs, **common_kwargs)

        self.netD_A = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.netD_A)
        self.netD_B = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.netD_B)

    def initialize_weights(self, first_train):
        if (first_train == True):
            self.netG_A2B.apply(weights_init_normal)
            self.netG_B2A.apply(weights_init_normal)
            self.netD_A.apply(weights_init_normal)
            self.netD_B.apply(weights_init_normal)
            epoch = 0
            loss_G = 0
            loss_D_A = 0
            loss_D_B = 0
        else:
            checkpointG_A2B = torch.load('netG_A2B.pt')
            self.netG_A2B.load_state_dict(checkpointG_A2B["netG_A2B_state_dict"])
            checkpointG_B2A = torch.load('netG_B2A.pt')
            self.netG_B2A.load_state_dict(checkpointG_B2A["netG_B2A_state_dict"])

            epoch = checkpointG_A2B["epoch"]
            loss_G = checkpointG_A2B["loss_G"]

            checkpointD_A = torch.load('netD_A.pt')
            self.netD_A.load_state_dict(checkpointD_A["netD_A_state_dict"])
            epoch = checkpointD_A["epoch"]
            loss_D_A = checkpointD_A["loss_D_A"]

            checkpointD_B = torch.load('netD_B.pt')
            self.netD_B.load_state_dict(checkpointD_B["netD_B_state_dict"])
            epoch = checkpointD_B["epoch"]
            loss_D_B = checkpointG_A2B["loss_D_B"]

        return epoch, loss_G, loss_D_A, loss_D_B


    def forward(self, real_A, real_B, fake_A_buffer, fake_B_buffer):

        same_B = self.netG_A2B.forward(real_B)
        same_A = self.netG_B2A.forward(real_A)

        # GENERATOR
        fake_B = self.netG_A2B.forward(real_A)
        pred_fakeB_GAN = self.netD_B.forward(fake_B)

        fake_A = self.netG_B2A.forward(real_B)
        pred_fakeA_GAN= self.netD_A.forward(fake_A)

        # Cycle loss
        recovered_A = self.netG_B2A.forward(fake_B)
        recovered_B = self.netG_A2B.forward(fake_A)

        # Real loss
        pred_realA_DIS = self.netD_A.forward(real_A.detach())
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fakeA_DIS = self.netD_A.forward(fake_A.detach())

        # Real loss
        pred_realB_DIS = self.netD_B.forward(real_B.detach())
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fakeB_DIS = self.netD_B.forward(fake_B.detach())

        return fake_A , fake_B, same_B, same_A, pred_fakeB_GAN, pred_fakeA_GAN, recovered_A, recovered_B, pred_realA_DIS, pred_fakeA_DIS, pred_realB_DIS, pred_fakeB_DIS