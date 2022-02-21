import argparse
import itertools
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional

from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

from StyleGAN2 import Discriminator
from StyleGAN3 import Generator
from dataset import ImageDataset
from utils.utils import EasyDict
from utils.utils import LambdaLR
from utils.utils import ReplayBuffer
from utils.utils import weights_init_normal
from utils.utils import Logger
import torchvision.models as models

# for distributed training
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.sync_batchnorm.batchnorm import convert_model

def main():
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
    parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')

    #StyleGAN3 parameters
    parser.add_argument('--cfg', help='Base configuration, possible choices: stylegan3-t, stylegan3-r,stylegan2',
                        type=str,
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
    parser.add_argument('--num_channels', help='Number of channels of the data, so the image.shape[0]', type=int,
                        default=3)
    parser.add_argument('--first_train', default=False, action='store_true', help='first training cycle')
    parser.add_argument('--clip_value', default=5, type=int, help='value used to clip the gradient')

    # for distributed training
    parser.add_argument('--parallel', default=True, action='store_true', help='use parallel computation')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)

    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    # Initialize config.
    G_kwargs = EasyDict(z_dim=1000, w_dim=512,mapping_kwargs=EasyDict())
    D_kwargs = EasyDict(block_kwargs=EasyDict(), mapping_kwargs=EasyDict(), epilogue_kwargs=EasyDict())
    # Hyperparameters & settings.
    batch_size = opt.batchSize
    G_kwargs.channel_base = D_kwargs.channel_base = opt.cbase
    G_kwargs.channel_max = D_kwargs.channel_max = opt.cmax
    G_kwargs.mapping_kwargs.num_layers = 2 if opt.map_depth is None else opt.map_depth
    D_kwargs.block_kwargs.freeze_layers = opt.freezed
    D_kwargs.epilogue_kwargs.mbstd_group_size = opt.mbstd_group

    # Base configuration.
    G_kwargs.magnitude_ema_beta = 0.5 ** (batch_size / (20 * 1e3))
    if opt.cfg == 'stylegan3-r':
        G_kwargs.conv_kernel = 1  # Use 1x1 convolutions.
        G_kwargs.channel_base *= 2  # Double the number of feature maps.
        G_kwargs.channel_max *= 2
        G_kwargs.use_radial_filters = True  # Use radially symmetric downsampling filters.
    common_kwargs = dict(c_dim=opt.label_dim, img_resolution=opt.resolution, img_channels=opt.num_channels)

    # ##### Definition of variables ##### #

    # for distributed training
    if opt.parallel == True:
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        torch.distributed.init_process_group(backend="nccl")
        print(
            f"[{os.getpid()}] world_size = {torch.distributed.get_world_size()}, "
            + f"rank = {torch.distributed.get_rank()}, backend={torch.distributed.get_backend()}"
        )

    # Generators
    netG_A2B = Generator(**G_kwargs, **common_kwargs)
    netG_B2A = Generator(**G_kwargs, **common_kwargs)
    # Discriminators
    netD_A = Discriminator(**D_kwargs, **common_kwargs)
    netD_B = Discriminator(**D_kwargs, **common_kwargs)


    if (opt.cuda == True and opt.parallel == False):
        device = torch.device('cuda')
        netG_A2B.to(device)
        netG_B2A.to(device)
        netD_A.to(device)
        netD_B.to(device)

    if (opt.cuda == True and opt.parallel == True):
        n = torch.cuda.device_count() // opt.local_world_size
        device_ids = list(range(opt.local_rank * n, (opt.local_rank + 1) * n))

        print(
            f"[{os.getpid()}] rank = {torch.distributed.get_rank()}, "
            + f"world_size = {torch.distributed.get_world_size()}, n = {n}, device_ids = {device_ids}"
        )

        # BatchNorm Synchronization
        netD_A = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netD_A)
        netD_B = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netD_B)
        netG_A2B = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netG_A2B)
        netG_B2A = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netG_B2A)


        netG_A2B = netG_A2B.cuda(device_ids[0])
        netG_B2A = netG_B2A.cuda(device_ids[0])
        netD_A = netD_A.cuda(device_ids[0])
        netD_B = netD_B.cuda(device_ids[0])

        #for distributed training
        netD_A = DDP(netD_A, device_ids=device_ids, broadcast_buffers=False)
        netD_B = DDP(netD_B, device_ids=device_ids, broadcast_buffers=False)
        netG_A2B = DDP(netG_A2B, device_ids=device_ids, broadcast_buffers=False)
        netG_B2A = DDP(netG_B2A, device_ids=device_ids, broadcast_buffers=False)

    if (opt.first_train == True):
        #initialize weights
        netG_A2B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()


    #CAMBIARE IL LEARNING RATE E METTERNE DUE DIVERSI PER GENERATORE E DISCRIMINATORE.
    #DISCRIMINATORE: LR_D=0.002
    #GENERATORE: LR_G=0.0025

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    #fix the random seed
    torch.manual_seed(10)

    # Dataset loader
    transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(opt.size),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
    dataset = ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True)

    if opt.parallel == True:
        dist_sampler = DistributedSampler(dataset, rank=int(os.environ['RANK']), num_replicas=int(os.environ['WORLD_SIZE']))
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu, sampler=dist_sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    if (opt.first_train == False):
        checkpointG_A2B = torch.load('output/netG_A2B.pt')
        netG_A2B.load_state_dict(checkpointG_A2B["netG_A2B_state_dict"])
        checkpointG_B2A = torch.load('output/netG_B2A.pt')
        netG_B2A.load_state_dict(checkpointG_B2A["netG_B2A_state_dict"])
        optimizer_G.load_state_dict(checkpointG_A2B["optimizer_G_state_dict"])
        opt.epoch = checkpointG_A2B["epoch"]
        loss_G = checkpointG_A2B["loss_G"]

        checkpointD_A = torch.load('output/netD_A.pt')
        netD_A.load_state_dict(checkpointD_A["netD_A_state_dict"])
        optimizer_D_A.load_state_dict(checkpointD_A["optimizer_D_A_state_dict"])
        opt.epoch = checkpointD_A["epoch"]
        loss_D_A = checkpointD_A["loss_D_A"]

        checkpointD_B = torch.load('output/netD_B.pt')
        netD_B.load_state_dict(checkpointD_B["netD_B_state_dict"])
        optimizer_D_B.load_state_dict(checkpointD_B["optimizer_D_B_state_dict"])
        opt.epoch = checkpointD_B["epoch"]
        loss_D_B = checkpointD_B["loss_D_B"]


    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader), opt.epoch)

    #loading resnet18 pretrained
    resnet18 = models.resnet18(pretrained=True)
    resnet18.eval().cuda(device_ids[0])

    # ##### Training ######
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(opt.epoch, opt.n_epochs):
            for i, batch in enumerate(dataloader):
                # Set model input
                real_A = Variable(input_A.copy_(batch['A']))
                real_B = Variable(input_B.copy_(batch['B']))

                ###### Generators A2B and B2A ######
                netG_A2B.requires_grad_(True)
                netG_B2A.requires_grad_(True)
                netD_A.requires_grad_(False)
                netD_B.requires_grad_(False)

                optimizer_G.zero_grad()

                # Identity loss
                # G_A2B(B) should equal B if real B is fed
                with torch.no_grad():
                    z = resnet18(real_B)
                same_B = netG_A2B(z.cuda(device_ids[0]))
                loss_identity_B = criterion_identity(same_B, real_B) * 5.0
                # G_B2A(A) should equal A if real A is fed
                with torch.no_grad():
                    z = resnet18(real_A)
                same_A = netG_B2A(z.cuda(device_ids[0]))
                loss_identity_A = criterion_identity(same_A, real_A) * 5.0

                # GAN loss
                with torch.no_grad():
                    z = resnet18(real_A)
                fake_B = netG_A2B(z.cuda(device_ids[0]))
                with torch.no_grad():
                    pred_fake = netD_B(fake_B)
                loss_GAN_A2B = criterion_GAN(pred_fake.view(-1), target_real)
                with torch.no_grad():
                    z = resnet18(real_B)
                fake_A = netG_B2A(z.cuda(device_ids[0]))
                with torch.no_grad():
                    pred_fake = netD_A(fake_A)
                loss_GAN_B2A = criterion_GAN(pred_fake.view(-1), target_real)

                # Cycle loss
                with torch.no_grad():
                    z = resnet18(fake_B)
                recovered_A = netG_B2A(z.cuda(device_ids[0]))
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

                with torch.no_grad():
                    z = resnet18(fake_A)
                recovered_B = netG_A2B(z.cuda(device_ids[0]))
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

                # Total loss
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(netG_A2B.parameters(), opt.clip_value)
                torch.nn.utils.clip_grad_norm_(netG_B2A.parameters(), opt.clip_value)

                optimizer_G.step()
                ###################################

                # ##### Discriminator A ######
                netG_A2B.requires_grad_(False)
                netG_B2A.requires_grad_(False)
                netD_A.requires_grad_(True)
                netD_B.requires_grad_(False)

                optimizer_D_A.zero_grad()

                # Real loss
                pred_real = netD_A(real_A)
                loss_D_real = criterion_GAN(pred_real.view(-1), target_real)

                # Fake loss
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = netD_A(fake_A.detach())
                loss_D_fake = criterion_GAN(pred_fake.view(-1), target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake) * 0.5

                loss_D_A.backward()
                torch.nn.utils.clip_grad_norm_(netD_A.parameters(), opt.clip_value)
                optimizer_D_A.step()
                ###################################

                # ##### Discriminator B ######
                netG_A2B.requires_grad_(False)
                netG_B2A.requires_grad_(False)
                netD_A.requires_grad_(False)
                netD_B.requires_grad_(True)

                optimizer_D_B.zero_grad()

                # Real loss
                pred_real = netD_B(real_B)
                loss_D_real = criterion_GAN(pred_real.view(-1), target_real)

                # Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B.detach())
                loss_D_fake = criterion_GAN(pred_fake.view(-1), target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake) * 0.5

                loss_D_B.backward()
                torch.nn.utils.clip_grad_norm_(netD_B.parameters(), opt.clip_value)
                optimizer_D_B.step()
                ###################################

                logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
                            'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                           images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

            # Save models checkpoints for test
            torch.save(netG_A2B.state_dict(), 'netG_A2B.pth')
            torch.save(netG_B2A.state_dict(), 'netG_B2A.pth')
            torch.save(netD_A.state_dict(), 'netD_A.pth')
            torch.save(netD_B.state_dict(), 'netD_B.pth')

            # Save models checkpoints for resume the training
            torch.save({"netG_A2B_state_dict": netG_A2B.state_dict(), "epoch": epoch,
                        "optimizer_G_state_dict": optimizer_G.state_dict(), "loss_G": loss_G}, 'output/netG_A2B.pt')
            torch.save({"netG_B2A_state_dict": netG_B2A.state_dict(), "epoch": epoch,
                        "optimizer_G_state_dict": optimizer_G.state_dict(), "loss_G": loss_G}, 'output/netG_B2A.pt')
            torch.save({"netD_A_state_dict": netD_A.state_dict(), "epoch": epoch,
                        "optimizer_D_A_state_dict": optimizer_D_A.state_dict(), "loss_D_A": loss_D_A},
                       'output/netD_A.pt')
            torch.save({"netD_B_state_dict": netD_B.state_dict(), "epoch": epoch,
                        "optimizer_D_B_state_dict": optimizer_D_B.state_dict(), "loss_D_B": loss_D_B},
                       'output/netD_B.pt')
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()