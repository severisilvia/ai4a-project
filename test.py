import argparse
import itertools
import os
import sys

import torch
import torchvision.transforms as transforms
import torch.nn.functional

from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from utils.utils import tensor2image

from StyleGAN2 import Discriminator
from StyleGAN3 import Generator
from dataset import ImageDataset
from utils.utils import EasyDict
from utils.utils import LambdaLR
from utils.utils import ReplayBuffer
from utils.utils import weights_init_normal
from utils.utils import Logger
import torchvision.models as models



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/day_night', help='root directory of the datasets')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=512, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', default=True, action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='netG_A2B.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='netG_B2A.pth', help='B2A generator checkpoint file')

    # StyleGAN3 parameters
    parser.add_argument('--cfg', help='Base configuration, possible choices: stylegan3-t, stylegan3-r,stylegan2',
                        type=str, default='stylegan3-t' )
    parser.add_argument('--cbase', help='Capacity multiplier', type=int, default=32768)
    parser.add_argument('--cmax',  help='Max. feature maps', type=int, default=512)
    parser.add_argument('--map-depth', help='Mapping network depth  [default: varies]', type=int, default=2)
    parser.add_argument('--freezed', help='Freeze first layers of D', type=int, default=0)
    parser.add_argument('--mbstd-group', help='Minibatch std group size', type=int, default=4)
    parser.add_argument('--label_dim', help='Number of labels', type=int, default=0)
    parser.add_argument('--resolution', help='Resolution of the images expressed as the dimension of one of the'
                                             ' two equals dimension image.shape[1] or image.shape[2] of the image, '
                                             'note that we want squared images obviously', type=int, default=512)
    parser.add_argument('--num_channels', help='Number of channels of the data, so the image.shape[0]', type=int, default=3)
    parser.add_argument('--parallel', default=False, action='store_true', help='use parallel computation')

    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # ##### Definition of variables ######

    # Costruzione argomenti per istanziare modelli
    # Initialize config.
    G_kwargs = EasyDict(z_dim=1000, w_dim=512, mapping_kwargs=EasyDict())
    #class_name = None
    # Hyperparameters & settings.
    batch_size = opt.batchSize
    G_kwargs.channel_base = opt.cbase
    G_kwargs.channel_max = opt.cmax
    G_kwargs.mapping_kwargs.num_layers = 2 if opt.map_depth is None else opt.map_depth
    # metrics = opts.metrics
    # Base configuration.
    #G_kwargs.class_name = 'training.networks_stylegan3.Generator'
    G_kwargs.magnitude_ema_beta = 0.5 ** (batch_size / (20 * 1e3))
    if opt.cfg == 'stylegan3-r':
        G_kwargs.conv_kernel = 1  # Use 1x1 convolutions.
        G_kwargs.channel_base *= 2  # Double the number of feature maps.
        G_kwargs.channel_max *= 2
        G_kwargs.use_radial_filters = True  # Use radially symmetric downsampling filters.
    common_kwargs = dict(c_dim=opt.label_dim, img_resolution=opt.resolution, img_channels=opt.num_channels)

    # Networks
    netG_A2B = Generator(**G_kwargs, **common_kwargs)
    netG_B2A = Generator(**G_kwargs, **common_kwargs)
    resnet18 = models.resnet18(pretrained=True)

    #print(torch.load(opt.generator_A2B).keys())
    #print(torch.load(opt.generator_B2A).keys())

    state_dictG_A2B=torch.load(opt.generator_A2B)
    for key in list(state_dictG_A2B.keys()):
        state_dictG_A2B[key.replace('module.', '')] = state_dictG_A2B.pop(key)
    #print(state_dictG_A2B.keys())

    state_dictG_B2A = torch.load(opt.generator_B2A)
    for key in list(state_dictG_B2A.keys()):
        state_dictG_B2A[key.replace('module.', '')] = state_dictG_B2A.pop(key)
    #print(state_dictG_B2A.keys())


    if opt.cuda:
        device = torch.device('cuda')
        netG_A2B.to(device)
        netG_B2A.to(device)
        resnet18.to(device)


    # Load state dicts
    print("load state dict")
    netG_A2B.load_state_dict(state_dictG_A2B)
    netG_B2A.load_state_dict(state_dictG_B2A)

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()
    resnet18.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    # Dataset loader
    transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(opt.size),
                   transforms.ToTensor(),
                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]

    dataset=ImageDataset(opt.dataroot, transforms_=transforms_, mode='test')
    dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    ###################################

    ###### Testing######

    # Create output dirs if they don't exist
    if not os.path.exists('output/A'):
        print('creating output/A...')
        os.makedirs('output/A')
    if not os.path.exists('output/B'):
        print('creating output/B...')
        os.makedirs('output/B')

    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Generate output
        fake_A_boh = netG_B2A(resnet18(real_B)).data
        fake_B_boh = netG_A2B(resnet18(real_A)).data
        fake_B = 0.5*(netG_A2B(resnet18(real_A)).data + 1.0)
        fake_A = 0.5*(netG_B2A(resnet18(real_B)).data + 1.0)

        image_to_print = fake_A
        plt.imshow(tensor2image(image_to_print.detach()).transpose((1, 2, 0)))
        plt.show()
        image_to_print = fake_B
        plt.imshow(tensor2image(image_to_print.detach()).transpose((1, 2, 0)))
        plt.show()


        # Save image files
        save_image(fake_A, 'output/A/%04d.png' % (i+1))
        save_image(fake_B, 'output/B/%04d.png' % (i+1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')

if __name__ == '__main__':
    my_env = os.environ.copy()
    my_env["PATH"] = "/homes/sseveri/.conda/envs/stylegan3/bin:" + my_env["PATH"]
    os.environ.update(my_env)
    main()