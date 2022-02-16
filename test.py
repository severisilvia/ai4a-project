import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torchvision.models as models

from dataset import ImageDataset
from StyleGAN3 import Generator

from utils.utils import EasyDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/day_night', help='root directory of the datasets')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=512, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', default=True, action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')

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

    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # ##### Definition of variables ######

    # Costruzione argomenti per istanziare modelli
    # Initialize config.
    G_kwargs = EasyDict(class_name=None, z_dim=1000, w_dim=512, mapping_kwargs=EasyDict())
    # Hyperparameters & settings.
    batch_size = opt.batchSize
    G_kwargs.channel_base = opt.cbase
    G_kwargs.channel_max = opt.cmax
    G_kwargs.mapping_kwargs.num_layers = 2 if opt.map_depth is None else opt.map_depth
    # metrics = opts.metrics
    # Base configuration.
    G_kwargs.class_name = 'training.networks_stylegan3.Generator'
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


    if opt.cuda:
        device = torch.device('cuda')
        netG_A2B.to(device)
        netG_B2A.to(device)
        resnet18.to(device)

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()
    resnet18.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    # Dataset loader
    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
    ###################################

    ###### Testing######

    # Create output dirs if they don't exist
    if not os.path.exists('output/A'):
        os.makedirs('output/A')
    if not os.path.exists('output/B'):
        os.makedirs('output/B')

    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Generate output
        fake_B = 0.5*(netG_A2B(resnet18(real_A)).data + 1.0)
        fake_A = 0.5*(netG_B2A(resnet18(real_B)).data + 1.0)

        # Save image files
        save_image(fake_A, 'output/A/%04d.png' % (i+1))
        save_image(fake_B, 'output/B/%04d.png' % (i+1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')