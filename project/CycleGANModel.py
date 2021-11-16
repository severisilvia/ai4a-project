import itertools
import torch
import os
from torch import nn

from project import StyleGAN3
from project.StyleGAN3 import StyleGAN3_Generator, StyleGAN3_Discriminator
from project.utils.image_pool import ImagePool



class CycleGANModel():

    """
    Generators: G_A: A -> B; G_B: B -> A.
    Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A
    adversarial loss: style3gan loss --> questa parte è ancora da vedere
    Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A||
    Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B||
    cycle loss: forward cycle loss + backward cycle loss
    final loss: adversarial loss_A + aversarial loss_b + cycle loss

    """

#RIVEDI ALCUNE COSE DEL METODO INIT

    def __init__(self, opt): #credo che opt siano parametri opzionali, quando istanzio la classe posso passarli o meno
        super(CycleGANModel, self).__init__()

        #cerca di fare questo senza opt
        self.opt=opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.optimizers = []
        self.image_paths = []


        #define generators
        self.G_A = StyleGAN3_Generator()
        self.G_B = StyleGAN3_Generator()

        if self.isTrain:  # define discriminators
            self.D_A = StyleGAN3_Discriminator()
            self.D_B = StyleGAN3_Discriminator()

        if self.isTrain:

            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = StyleGAN3.GANLoss().to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()



            #questa roba non l'ho capita bene
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input): #RIFAI QUESTA COERENTE CON LA NOSTRA POLITICA DI GESTIONE DELL'INPUT E IL NOSTRO DATASET
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.G_A(self.real_A)  # G_A(A)
        self.rec_A = self.G_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.G_B(self.real_B)  # G_B(B)
        self.rec_B = self.G_A(self.fake_A)  # G_A(G_B(B))

