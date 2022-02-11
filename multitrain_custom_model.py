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
from custom_model import custom_model

# per training distribuito
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


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
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    parser.add_argument('--first_train', default=True, action='store_true', help='first training cycle')
    # cose per training distribuito
    parser.add_argument('--parallel', default=True, action='store_true', help='use parallel computation')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)

    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    #custom model-->wrapper
    model= custom_model(opt.batchSize)

    #initialize the model
    opt.ephoc, loss_G, loss_D_A, loss_D_B = model.initialize_weights(opt.first_train)

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(model.netG_A2B.parameters(), model.netG_B2A.parameters()),
                                       lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(model.netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(model.netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                           lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                              opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                             lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                             lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                opt.decay_epoch).step)
    # per training distribuito (DISTRIBUTED DATAPARALLEL)
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

    if(opt.first_train==False):
        checkpointG_A2B = torch.load('netG_A2B.pt')
        optimizer_G.load_state_dict(checkpointG_A2B["optimizer_G_state_dict"])
        checkpointD_A = torch.load('netD_A.pt')
        optimizer_D_A.load_state_dict(checkpointD_A["optimizer_D_A_state_dict"])
        checkpointD_B = torch.load('netD_B.pt')
        optimizer_D_B.load_state_dict(checkpointD_B["optimizer_D_B_state_dict"])


    if (opt.cuda == True and opt.parallel == False):
        device = torch.device('cuda')
        model.to(device)

    if (opt.cuda == True and opt.parallel == True):
        n = torch.cuda.device_count() // opt.local_world_size
        device_ids = list(range(opt.local_rank * n, (opt.local_rank + 1) * n))

        print(
            f"[{os.getpid()}] rank = {torch.distributed.get_rank()}, "
            + f"world_size = {torch.distributed.get_world_size()}, n = {n}, device_ids = {device_ids}"
        )

        model.cuda(device_ids[0])
        model_ddp = DDP(model, device_ids)


        #NON FUNZIONA, CAPISCI COSA FARE

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()


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

        dist_sampler = DistributedSampler(dataset, rank=int(os.environ['RANK']),
                                          num_replicas=int(os.environ['WORLD_SIZE']))
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu,
                                sampler=dist_sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader))

    # ##### Training ######
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(opt.epoch, opt.n_epochs):
            for i, batch in enumerate(dataloader):
                # Set model input
                real_A = Variable(input_A.copy_(batch['A']))
                real_B = Variable(input_B.copy_(batch['B']))

                ###### Generators A2B and B2A ######
                optimizer_G.zero_grad()

                fake_A, fake_B, same_B, same_A, pred_fakeB_GAN, pred_fakeA_GAN, recovered_A, recovered_B, pred_realA_DIS, pred_fakeA_DIS, pred_realB_DIS, pred_fakeB_DIS= model_ddp.forward(real_A, real_B, fake_A_buffer, fake_B_buffer)

                # Identity loss
                loss_identity_B = criterion_identity(same_B, real_B) * 5.0
                loss_identity_A = criterion_identity(same_A, real_A) * 5.0

                # GAN loss
                loss_GAN_A2B = criterion_GAN(pred_fakeB_GAN.view(-1), target_real)
                loss_GAN_B2A = criterion_GAN(pred_fakeA_GAN.view(-1), target_real)

                # Cycle loss
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

                # Total loss
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                loss_G.backward()

                optimizer_G.step()
                ###################################

                # ##### Discriminator A ######
                optimizer_D_A.zero_grad()

                # Real loss
                loss_D_real = criterion_GAN(pred_realA_DIS.view(-1), target_real)

                # Fake loss
                loss_D_fake = criterion_GAN(pred_fakeA_DIS.view(-1), target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake) * 0.5
                loss_D_A.backward()

                optimizer_D_A.step()
                ###################################

                # ##### Discriminator B ######
                optimizer_D_B.zero_grad()

                # Real loss
                loss_D_real = criterion_GAN(pred_realB_DIS.view(-1), target_real)

                # Fake loss
                loss_D_fake = criterion_GAN(pred_fakeB_DIS.view(-1), target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake) * 0.5
                loss_D_B.backward()

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

            # Save models checkpoints
            torch.save(model.netG_A2B.state_dict(), 'netG_A2B.pth')
            torch.save(model.netG_B2A.state_dict(), 'netG_B2A.pth')
            torch.save(model.netD_A.state_dict(), 'netD_A.pth')
            torch.save(model.netD_B.state_dict(), 'netD_B.pth')

            torch.save({"netG_A2B_state_dict": model.netG_A2B.state_dict(), "epoch": epoch,
                        "optimizer_G_state_dict": optimizer_G.state_dict(), "loss_G": loss_G}, 'output/netG_A2B.pt')
            torch.save({"netG_B2A_state_dict": model.netG_B2A.state_dict(), "epoch": epoch,
                        "optimizer_G_state_dict": optimizer_G.state_dict(), "loss_G": loss_G}, 'output/netG_B2A.pt')
            torch.save({"netD_A_state_dict": model.netD_A.state_dict(), "epoch": epoch,
                        "optimizer_D_A_state_dict": optimizer_D_A.state_dict(), "loss_D_A": loss_D_A},
                       'output/netD_A.pt')
            torch.save({"netD_B_state_dict": model.netD_B.state_dict(), "epoch": epoch,
                        "optimizer_D_B_state_dict": optimizer_D_B.state_dict(), "loss_D_B": loss_D_B},
                       'output/netD_B.pt')

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()