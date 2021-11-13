import torch
import os

class CycleGANModel():
    """Initialize the BaseModel class.
            Parameters:
                opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
            When creating your custom class, you need to implement your own initialization.
            In this function, you should first call <BaseModel.__init__(self, opt)>
            Then, you need to define four lists:
                -- self.loss_names (str list):          specify the training losses that you want to plot and save.
                -- self.model_names (str list):         define networks used in our training.
                -- self.visual_names (str list):        specify the images that you want to display and save.
                -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.

    Add new dataset-specific options, and rewrite default values for existing options.
            Parameters:
                parser          -- original option parser
                is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
            Returns:
                the modified parser.
            For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
            A (source domain), B (target domain).
            Generators: G_A: A -> B; G_B: B -> A.
            Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
            Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
            Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
            Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
            Dropout is not used in the original CycleGAN paper.
    """


    def __init__(self, opt):
        super(CycleGANModel, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
        'cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

