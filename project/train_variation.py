import argparse
import itertools

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

from StyleGAN2 import Discriminator
from StyleGAN3_variation import Generator
from dataset import ImageDataset
# Per costruire i dizionari da mandare come argomenti del training
from utils.utils import EasyDict
from utils.utils import LambdaLR
# cerca di capire queste funzioni a cosa servono
from utils.utils import ReplayBuffer
from utils.utils import weights_init_normal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='../datasets/day_night', help='root directory of the datasets')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')

    # Parsing roba per StyleGAN3
    parser.add_argument('--cfg', help='Base configuration, possible choices: stylegan3-t, stylegan3-r,stylegan2',
                        type=str,
                        default='stylegan3-t')
    parser.add_argument('--cbase', help='Capacity multiplier', type=int, default=32768)
    parser.add_argument('--cmax', help='Max. feature maps', type=int, default=512)
    parser.add_argument('--map-depth', help='Mapping network depth  [default: varies]', type=int, default=2)
    parser.add_argument('--freezed', help='Freeze first layers of D', type=int, default=0)
    parser.add_argument('--mbstd-group', help='Minibatch std group size', type=int, default=4)
    parser.add_argument('--label_dim', help='Number of labels', type=int, default=2)
    parser.add_argument('--resolution',
                        help='Resolution of the images expressed as the dimension of one of the two equals dimension image.shape[1] or image.shape[2] of the image, note that we want squared images obviously',
                        type=int, default=512)
    parser.add_argument('--num_channels', help='Number of channels of the data, so the image.shape[0]', type=int,
                        default=3)
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Costruzione argomenti per istanziare modelli
    # Initialize config.
    G_kwargs = EasyDict(img_real_dim=512)
    D_kwargs = EasyDict(block_kwargs=EasyDict(), epilogue_kwargs=EasyDict())
    # Hyperparameters & settings.
    batch_size = opt.batchSize
    G_kwargs.channel_base = D_kwargs.channel_base = opt.cbase
    G_kwargs.channel_max = D_kwargs.channel_max = opt.cmax
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

    # ##### Definition of variables ##### #

    # Generators
    """     
            FOR THE GENERATOR:
            z_dim,                      # Input latent (Z) dimensionality.
            c_dim,                      # Conditioning label (C) dimensionality.
            w_dim,                      # Intermediate latent (W) dimensionality.
            img_resolution,             # Output resolution.
            img_channels,               # Number of output color channels.
            mapping_kwargs      = {
                    z_dim,                      # Input latent (Z) dimensionality.
                    c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
                    w_dim,                      # Intermediate latent (W) dimensionality.
                    num_ws,                     # Number of intermediate latents to output.
                    num_layers      = 2,        # Number of mapping layers.
                    lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
                    w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training.
                },   # Arguments for MappingNetwork. Di default è vuoto {}
            **synthesis_kwargs = {      #arguments for SynthesisNetwork
                    channel_base        = 32768,    # Overall multiplier for the number of channels.
                    channel_max         = 512,      # Maximum number of channels in any layer.
                    num_layers          = 14,       # Total number of layers, excluding Fourier features and ToRGB.
                    num_critical        = 2,        # Number of critically sampled layers at the end.
                    first_cutoff        = 2,        # Cutoff frequency of the first layer (f_{c,0}).
                    first_stopband      = 2**2.1,   # Minimum stopband of the first layer (f_{t,0}).
                    last_stopband_rel   = 2**0.3,   # Minimum stopband of the last layer, expressed relative to the cutoff.
                    margin_size         = 10,       # Number of additional pixels outside the image.
                    output_scale        = 0.25,     # Scale factor for the output image.
                    num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
                    **layer_kwargs = {              # Arguments for SynthesisLayer.
                            is_torgb,                       # Is this the final ToRGB layer?
                            is_critically_sampled,          # Does this layer use critical sampling?
                            use_fp16,                       # Does this layer use FP16?

                            # Input & output specifications.
                            in_channels,                    # Number of input channels.
                            out_channels,                   # Number of output channels.
                            in_size,                        # Input spatial size: int or [width, height].
                            out_size,                       # Output spatial size: int or [width, height].
                            in_sampling_rate,               # Input sampling rate (s).
                            out_sampling_rate,              # Output sampling rate (s).
                            in_cutoff,                      # Input cutoff frequency (f_c).
                            out_cutoff,                     # Output cutoff frequency (f_c).
                            in_half_width,                  # Input transition band half-width (f_h).
                            out_half_width,                 # Output Transition band half-width (f_h).

                            # Hyperparameters.
                            conv_kernel         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
                            filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
                            lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
                            use_radial_filters  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
                            conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
                            magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes.
                }
    """

    netG_A2B = Generator(**G_kwargs, **common_kwargs)
    netG_B2A = Generator(**G_kwargs, **common_kwargs)

    # Discriminators
    """     
            FOR THE DISCRIMINATOR:
            c_dim,                          # Conditioning label (C) dimensionality. Nel codice lo inizializza così --> c_dim=training_set.label_dim
            img_resolution,                 # Input resolution.
            img_channels,                   # Number of input color channels.
            architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
            channel_base        = 32768,    # Overall multiplier for the number of channels.
            channel_max         = 512,      # Maximum number of channels in any layer.
            num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
            conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
            cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
            block_kwargs        = {},       # Arguments for DiscriminatorBlock.
            mapping_kwargs      = {},       # Arguments for MappingNetwork.
            epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    """
    netD_A = Discriminator(**D_kwargs, **common_kwargs)
    netD_B = Discriminator(**D_kwargs, **common_kwargs)

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers --> CAPISCI COSA FANNO
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                          opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                         lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                            opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                         lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                            opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(opt.size),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    robo = ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True)
    dataloader = DataLoader(robo,
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    # Loss plot
    # logger = Logger(opt.n_epochs, len(dataloader))

    # ##### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()


            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()
            ###################################

            # ##### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            # ##### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            # Progress report (http://localhost:8097)
            # logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
            #           'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
            #          'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
            #        images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), 'netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'netD_A.pth')
        torch.save(netD_B.state_dict(), 'netD_B.pth')

        torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'output/netD_A.pth')
        torch.save(netD_B.state_dict(), 'output/netD_B.pth')

