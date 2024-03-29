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

# per training distribuito
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


if __name__ == '__main__':
    my_env = os.environ.copy()
    my_env["PATH"] = "/homes/sseveri/.conda/envs/stylegan3/bin:" + my_env["PATH"]
    os.environ.update(my_env)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/day_night', help='root directory of the datasets')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', default=True, action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')

    # Parsing roba per StyleGAN3
    parser.add_argument('--cfg', help='Base configuration, possible choices: stylegan3-t, stylegan3-r,stylegan2', type=str,
                        default='stylegan3-t')
    parser.add_argument('--cbase', help='Capacity multiplier', type=int, default=32768)
    parser.add_argument('--cmax', help='Max. feature maps', type=int, default=512)
    parser.add_argument('--map-depth', help='Mapping network depth  [default: varies]', type=int, default=2)
    parser.add_argument('--freezed', help='Freeze first layers of D', type=int, default=0)
    parser.add_argument('--mbstd-group', help='Minibatch std group size', type=int, default=4)
    parser.add_argument('--label_dim', help='Number of labels', type=int, default=0)
    parser.add_argument('--resolution',
                        help='Resolution of the images expressed as the dimension of one of the two equals dimension image.shape[1] or image.shape[2] of the image, note that we want squared images obviously',
                        type=int, default=512)
    parser.add_argument('--num_channels', help='Number of channels of the data, so the image.shape[0]', type=int, default=3)
    parser.add_argument('--first_train', default=True, action='store_true', help='first training cycle')

    #cose per training distribuito
    parser.add_argument('--parallel', default=False, action='store_true', help='use parallel computation')
    #(DATAPARALLEL)
    #parser.add_argument('--gpus', type=str, default='0,1,2', help='gpuids eg: 0,1,2,3')
    #(DISTRIBUTED DATAPARALLEL)
    parser.add_argument("--local_rank", default=0, type=int)

    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Costruzione argomenti per istanziare modelli
    # Initialize config.
    G_kwargs = EasyDict(z_dim=1000, w_dim=512, mapping_kwargs=EasyDict()) #potremmo aumentare per entrare ed uscire dal generatore con più feature!
    D_kwargs = EasyDict(block_kwargs=EasyDict(), mapping_kwargs=EasyDict(), epilogue_kwargs=EasyDict())
    # Hyperparameters & settings.
    batch_size = opt.batchSize
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

    # ##### Definition of variables ##### #

    #per training distribuito(DATAPARALLEL)
    #gpus = [int(i) for i in opt.gpus.split(',')]

    #per training distribuito (DISTRIBUTED DATAPARALLEL)
    if opt.parallel == True:
        torch.distributed.init_process_group('nccl', rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))

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

    if (opt.cuda == True and opt.parallel == False):
        device = torch.device('cuda')
        netG_A2B.to(device)
        netG_B2A.to(device)
        netD_A.to(device)
        netD_B.to(device)
    if (opt.cuda == True and opt.parallel == True):
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        print(device)
        netG_A2B.to(device)
        netG_B2A.to(device)
        netD_A.to(device)
        netD_B.to(device)
        #(DATAPARALLEL)
        # netD_A = torch.nn.DataParallel(netD_A, device_ids=gpus)
        # netD_B = torch.nn.DataParallel(netD_B, device_ids=gpus)
        # netG_A2B = torch.nn.DataParallel(netG_A2B, device_ids=gpus)
        # netG_B2A = torch.nn.DataParallel(netG_B2A, device_ids=gpus)
        #(DISTRIBUTED DATAPARALLEL)
        netD_A = DDP(netD_A, device_ids=[int(os.environ['LOCAL_RANK'])], output_device=int(os.environ['LOCAL_RANK']))
        netD_B = DDP(netD_B, device_ids=[int(os.environ['LOCAL_RANK'])], output_device=int(os.environ['LOCAL_RANK']))
        netG_A2B = DDP(netG_A2B, device_ids=[int(os.environ['LOCAL_RANK'])], output_device=int(os.environ['LOCAL_RANK']))
        netG_B2A = DDP(netG_B2A, device_ids=[int(os.environ['LOCAL_RANK'])], output_device=int(os.environ['LOCAL_RANK']))

    if(opt.first_train==True):
        netG_A2B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()  # VEDI SE VA BENE
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                         lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                         lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

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
    dataset = ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True)
    if opt.parallel == True:
        #(DISTRIBUTED DATAPARALLEL)
        dist_sampler = DistributedSampler(dataset, rank=int(os.environ['RANK']), num_replicas=int(os.environ['WORLD_SIZE']))
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu, sampler=dist_sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader))

    if (opt.first_train == False):
        checkpointG_A2B= torch.load('netG_A2B.pt')
        netG_A2B.load_state_dict(checkpointG_A2B["netG_A2B_state_dict"])
        checkpointG_B2A = torch.load('netG_B2A.pt')
        netG_B2A.load_state_dict(checkpointG_B2A["netG_B2A_state_dict"])
        optimizer_G.load_state_dict(checkpointG_A2B["optimizer_G_state_dict"])
        opt.epoch=checkpointG_A2B["epoch"]
        loss_G =checkpointG_A2B["loss_G"]

        checkpointD_A = torch.load('netD_A.pt')
        netD_A.load_state_dict(checkpointD_A["netD_A_state_dict"])
        optimizer_D_A.load_state_dict(checkpointD_A["optimizer_D_A_state_dict"])
        opt.epoch = checkpointD_A["epoch"]
        loss_D_A = checkpointD_A["loss_D_A"]

        checkpointD_B = torch.load('netD_B.pt')
        netD_B.load_state_dict(checkpointD_B["netD_B_state_dict"])
        optimizer_D_B.load_state_dict(checkpointD_B["optimizer_D_B_state_dict"])
        opt.epoch = checkpointD_B["epoch"]
        loss_D_B = checkpointG_A2B["loss_D_B"]

    # ##### Training ######
    with torch.autograd.set_detect_anomaly(True):
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
                loss_GAN_A2B = criterion_GAN(pred_fake.view(-1), target_real)

                fake_A = netG_B2A(real_B)
                pred_fake = netD_A(fake_A)
                loss_GAN_B2A = criterion_GAN(pred_fake.view(-1), target_real)

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
                loss_D_real = criterion_GAN(pred_real.view(-1), target_real)

                # Fake loss
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = netD_A(fake_A)
                loss_D_fake = criterion_GAN(pred_fake.view(-1), target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake) * 0.5
                loss_D_A.backward()

                optimizer_D_A.step()
                ###################################

                # ##### Discriminator B ######
                optimizer_D_B.zero_grad()

                # Real loss
                pred_real = netD_B(real_B)
                loss_D_real = criterion_GAN(pred_real.view(-1), target_real)

                # Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B)
                loss_D_fake = criterion_GAN(pred_fake.view(-1), target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake) * 0.5
                loss_D_B.backward()

                optimizer_D_B.step()
                ###################################

                # Progress report (http://localhost:8097)
                logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
                          'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                         'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                       images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})


                # print({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
                #            'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                #          'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)})
                # images = {'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B}
                #
                # image_to_print = real_A
                # plt.imshow(tensor2image(image_to_print.detach()).transpose((1, 2, 0)))
                # plt.show()

            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

            # Save models checkpoints
            torch.save(netG_A2B.state_dict(), 'netG_A2B.pth')
            torch.save(netG_B2A.state_dict(), 'netG_B2A.pth')
            torch.save(netD_A.state_dict(), 'netD_A.pth')
            torch.save(netD_B.state_dict(), 'netD_B.pth')

            torch.save({"netG_A2B_state_dict": netG_A2B.state_dict(), "epoch": epoch ,
                        "optimizer_G_state_dict": optimizer_G.state_dict() , "loss_G": loss_G  }, 'output/netG_A2B.pt')
            torch.save({"netG_B2A_state_dict": netG_B2A.state_dict(), "epoch": epoch,
                        "optimizer_G_state_dict": optimizer_G.state_dict(), "loss_G": loss_G}, 'output/netG_B2A.pt')
            torch.save({"netD_A_state_dict": netD_A.state_dict(), "epoch": epoch,
                        "optimizer_D_A_state_dict": optimizer_D_A.state_dict(), "loss_D_A": loss_D_A}, 'output/netD_A.pt')
            torch.save({"netD_B_state_dict": netD_B.state_dict(), "epoch": epoch,
                        "optimizer_D_B_state_dict": optimizer_D_B.state_dict(), "loss_D_B": loss_D_B}, 'output/netD_B.pt')

