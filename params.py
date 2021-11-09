'''
All the other python files import this file by default
to define the required parameters to create/load/initiate/train/test/save/log
the networks, models, logs and results
'''
from easydict import EasyDict as edict
import time
from os.path import join
opt_exp = edict()

# ---------- Global Experiment param --------------
opt_exp.isTrain = True #type=bool, default=True, help='enables backpropogation, else the network is only used for evlauation')
opt_exp.continue_train = False #type=bool, default=False, help='continue training: load the latest model')
opt_exp.starting_epoch_count = 0 #type=int, default=1, help='the starting epoch count, we save the model by <starting_epoch_count>, <starting_epoch_count>+<save_latest_freq>, ...')
opt_exp.save_latest_freq = 5000 #type=int, default=5000, help='frequency of saving the latest results')
opt_exp.save_epoch_freq = 1 #type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
opt_exp.n_epochs = 50 #type=int, default=50, help='# of Epochs to run the training for')
opt_exp.gpu_ids = ['1','2','3','0'] #type=tuple of char, default=['1','2','3','0'], help='gpu ids: e.g. ['0']  ['0','1','2'], ['0','2']. CPU implementation is not supported. gpu_ids[0] is used for loading the network and the rest for DataParellilization')
opt_exp.data = "rw_to_rw" #type=str, default='rw_to_rw', help='Dataset loader, switch case system [rw_to_rw|rw_to_rw_atk|rw_to_rw_env2|rw_to_rw_env3|rw_to_rw_env4|rw_to_rw_40|rw_to_rw_20|data_segment]')
opt_exp.n_decoders = 2 #type=int, default=2, help='# of Decoders to be used [1:Only Location Decoder|2:Both Location and Consistency Decoder]')

opt_exp.batch_size = 32 #type=int, default=32, help='batch size for training and testing the network')
opt_exp.ds_step_trn = 1 #type=int, default=1, help='data sub-sampling number for the training data')
opt_exp.ds_step_tst = 1 #type=int, default=1, help='data sub-sampling number for the testing data')
opt_exp.weight_decay = 1e-5 #type=float, default=1e-5, help='weight decay parameter for the Adam optimizer')

# ------ name of experiment ----------
opt_exp.save_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) # experiment name when train_and_test.py is ran
opt_exp.checkpoints_dir = join('./runs', opt_exp.save_name) # trained models are saved here
opt_exp.results_dir = opt_exp.checkpoints_dir # the resulting images from the offset decoder and the decoder are saved here
opt_exp.log_dir = opt_exp.checkpoints_dir # the logs of the median, 90th, 99th percentile errors, compensation ecoder and location decoder losses are saved for each epoch and each batch here
opt_exp.load_dir = opt_exp.checkpoints_dir # when loading a pre-trained model it is loaded from here

# ---------- offset encoder param --------------
opt_encoder = edict()
opt_encoder.parent_exp = opt_exp
opt_encoder.batch_size = opt_encoder.parent_exp.batch_size #type=int, default=1, help='input batch size')
opt_encoder.ngf = 64 #type=int, default=64, help='# of gen filters in first conv layer')
opt_encoder.base_model = 'resnet_encoder' #type=str, default='resnet_encoder', help='selects model to use for netG')
opt_encoder.net = 'G' #type=str, default='G', help='selects model to use for netG')
opt_encoder.resnet_blocks = 6 #type=int, default=6, help='# of resent blocks to use')
opt_encoder.no_dropout = False #type=bool, default=False help='no dropout for the generator')
opt_encoder.init_type = 'xavier' #type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
opt_encoder.init_gain = 0.02 #type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
opt_encoder.norm = 'instance' #type=str, default='instance', help='instance normalization or batch normalization')
opt_encoder.beta1 = 0.5 #type=float, default=0.5, help='momentum term of adam')
opt_encoder.lr = 0.00001 #type=float, default=0.0002, help='initial learning rate for adam')
opt_encoder.lr_policy = 'step' #type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
opt_encoder.lr_decay_iters = 50 #type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
opt_encoder.lambda_L = 1 #type=float, default=1, help='weightage given to the Generator')
opt_encoder.lambda_cross = 1e-5 #type=float, default=1e-4, help='weight for cross entropy loss')
opt_encoder.lambda_reg = 5e-4 #type=float, default=5e-4, help='regularization for the two encoder case')
opt_encoder.weight_decay = opt_encoder.parent_exp.weight_decay


opt_encoder.input_nc = 4 #type=int, default=3, help='# of input image channels')
opt_encoder.output_nc = 1 #type=int, default=3, help='# of output image channels')
opt_encoder.save_latest_freq = opt_encoder.parent_exp.save_latest_freq
opt_encoder.save_epoch_freq = opt_encoder.parent_exp.save_epoch_freq
opt_encoder.n_epochs = opt_encoder.parent_exp.n_epochs
opt_encoder.isTrain = opt_encoder.parent_exp.isTrain #type=bool, default=True, help='whether to train the network encoder or not')
opt_encoder.continue_train = False #type=bool, default=False, help='continue training: load the latest model')
opt_encoder.starting_epoch_count = opt_encoder.parent_exp.starting_epoch_count #type=int, default=1, help='the starting epoch count, we save the model by <starting_epoch_count>, <starting_epoch_count>+<save_latest_freq>, ...')
opt_encoder.name = 'encoder' #type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
opt_encoder.loss_type = "NoLoss" #type=string, default='NoLoss', help='Loss type for the network to enforce ['NoLoss'|'L1'|'L2'|'L1_sumL2'|'L2_sumL2'|'L2_sumL1'|'L2_offset_loss'|'L1_offset_loss'|'L1_sumL2_cross'|'L2_sumL2_cross']')
opt_encoder.niter = 20 #type=int, default=100, help='# of iter at starting learning rate')
opt_encoder.niter_decay = 100 #type=int, default=100, help='# of iter to linearly decay learning rate to zero')



opt_encoder.gpu_ids = opt_encoder.parent_exp.gpu_ids #type=tuple of char, default=['1','2','3','0'], help='gpu ids: e.g. ['0']  ['0','1','2'], ['0','2']. CPU implementation is not supported. gpu_ids[0] is used for loading the network and the rest for DataParellilization')
opt_encoder.num_threads = 4 #default=4, type=int, help='# threads for loading data')
opt_encoder.checkpoints_load_dir =  opt_encoder.parent_exp.load_dir #type=str, default='./checkpoints', help='models are saved here')
opt_encoder.checkpoints_save_dir =  opt_encoder.parent_exp.checkpoints_dir #type=str, default='./checkpoints', help='models are saved here')
opt_encoder.results_dir = opt_encoder.parent_exp.results_dir
opt_encoder.log_dir =  opt_encoder.parent_exp.log_dir #type=str, default='./checkpoints', help='models are saved here')
opt_encoder.max_dataset_size = float("inf") #type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
opt_encoder.verbose = False  #type=bool, default=False, help='if specified, print more debugging information')
opt_encoder.suffix ='' #default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')


# ---------- decoder param --------------
opt_decoder = edict()
opt_decoder.parent_exp = opt_exp
opt_decoder.batch_size = opt_decoder.parent_exp.batch_size #type=int, default=1, help='input batch size')
opt_decoder.ngf = 64 #type=int, default=64, help='# of gen filters in first conv layer')
opt_decoder.base_model = 'resnet_decoder' #type=str, default='resnet_decoder', help='selects model to use for netG')
opt_decoder.net = 'G' #type=str, default='G', help='selects model to use for netG')opt_decoder.no_dropout = False #type=bool, default=False, help='no dropout for the generator')
opt_decoder.resnet_blocks = 9 #type=int, default=9, help='total number of resent blocks including the ones in the encoder')
opt_decoder.encoder_res_blocks = opt_encoder.resnet_blocks
opt_decoder.no_dropout = False #type=bool, default=False, help='To not appply dropout layer')
opt_decoder.init_type = 'xavier' #type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
opt_decoder.init_gain = 0.02 #type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
opt_decoder.norm = 'instance' #type=str, default='instance', help='instance normalization or batch normalization')
opt_decoder.beta1 = 0.5 #type=float, default=0.5, help='momentum term of adam')
opt_decoder.lr = opt_encoder.lr  #type=float, default=0.0002, help='initial learning rate for adam')
opt_decoder.lr_policy = 'step' #type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
opt_decoder.lr_decay_iters = 20 #type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
opt_decoder.lambda_L = 1 #type=float, default=1, help='weightage given to the Generator')
opt_decoder.lambda_cross = 1e-5 #type=float, default=1e-5, help='weight given to cross entropy loss')
opt_decoder.lambda_reg = 5e-4 #type=float, default=5e-4, help='regularization weight')
opt_decoder.weight_decay = opt_decoder.parent_exp.weight_decay


opt_decoder.input_nc = 4 #type=int, default=4, help='# of input image channels')
opt_decoder.output_nc = 1 #type=int, default=1, help='# of output image channels')
opt_decoder.save_latest_freq = opt_decoder.parent_exp.save_latest_freq
opt_decoder.save_epoch_freq = opt_decoder.parent_exp.save_epoch_freq
opt_decoder.n_epochs = opt_decoder.parent_exp.n_epochs
opt_decoder.isTrain = opt_decoder.parent_exp.isTrain #type=bool, default=True, help='whether to train the network or not')
opt_decoder.continue_train = False #type=bool, default=False, help='continue training: load the latest model')
opt_decoder.starting_epoch_count = opt_decoder.parent_exp.starting_epoch_count #type=int, default=1, help='the starting epoch count, we save the model by <starting_epoch_count>, <starting_epoch_count>+<save_latest_freq>, ...')
# opt_decoder.phase = opt_decoder.parent_exp.phase #type=str, default='train', help='train, val, test, etc')
opt_decoder.name = 'decoder' #type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
opt_decoder.loss_type = "L2_sumL1" #type=string, default='L2_sumL1', help='Loss type for the netowkr to enforce ['NoLoss'|'L1'|'L2'|'L1_sumL2'|'L2_sumL2'|'L2_sumL1'|'L2_offset_loss'|'L1_offset_loss'|'L1_sumL2_cross'|'L2_sumL2_cross']')
opt_decoder.niter = 20 #type=int, default=100, help='# of iter at starting learning rate')
opt_decoder.niter_decay = 100 #type=int, default=100, help='# of iter to linearly decay learning rate to zero')



opt_decoder.gpu_ids = opt_decoder.parent_exp.gpu_ids #type=tuple of char, default=['1','2','3','0'], help='gpu ids: e.g. ['0']  ['0','1','2'], ['0','2']. CPU implementation is not supported. gpu_ids[0] is used for loading the network and the rest for DataParellilization')
opt_decoder.num_threads = 4 #default=4, type=int, help='# threads for loading data')
opt_decoder.checkpoints_load_dir =  opt_decoder.parent_exp.load_dir #type=str, default='./checkpoints', help='models are saved here')
opt_decoder.checkpoints_save_dir =  opt_decoder.parent_exp.checkpoints_dir #type=str, default='./checkpoints', help='models are saved here')
opt_decoder.results_dir = opt_decoder.parent_exp.results_dir
opt_decoder.log_dir =  opt_decoder.parent_exp.log_dir #type=str, default='./checkpoints', help='models are saved here')
opt_decoder.verbose = False #type=bool, default=False, help='if specified, print more debugging information')
opt_decoder.suffix ='' #default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')


# ---------- offset decoder param --------------
opt_offset_decoder = edict()
opt_offset_decoder.parent_exp = opt_exp
opt_offset_decoder.batch_size = opt_offset_decoder.parent_exp.batch_size #type=int, default=1, help='input batch size')
opt_offset_decoder.ngf = 64 #type=int, default=64, help='# of gen filters in first conv layer')
opt_offset_decoder.base_model = 'resnet_decoder' #type=str, default='resnet_decoder', help='selects model to use for netG')
opt_offset_decoder.net = 'G' #type=str, default='G', help='selects model to use for netG')
opt_offset_decoder.resnet_blocks = 12 #type=int, default=12, help='total number of resent blocks including the ones in the encoder')
opt_offset_decoder.encoder_res_blocks = opt_encoder.resnet_blocks
opt_offset_decoder.no_dropout = False #type=bool, default=False, help='To not appply dropout layer')
opt_offset_decoder.init_type = 'xavier' #type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
opt_offset_decoder.init_gain = 0.02 #type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
opt_offset_decoder.norm = 'instance' #type=str, default='instance', help='instance normalization or batch normalization')
opt_offset_decoder.beta1 = 0.5 #type=float, default=0.5, help='momentum term of adam')
opt_offset_decoder.lr = opt_encoder.lr #type=float, default=0.0002, help='initial learning rate for adam')
opt_offset_decoder.lr_policy = 'step' #type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
opt_offset_decoder.lr_decay_iters = 50 #type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
opt_offset_decoder.lambda_L = 1 # weightage given to the Generator
opt_offset_decoder.lambda_cross = 0 #type=float, default=1e-5, help='weight given to cross entropy loss')
opt_offset_decoder.lambda_reg = 0 #type=float, default=5e-4, help='regularization weight')
opt_offset_decoder.weight_decay = opt_offset_decoder.parent_exp.weight_decay


opt_offset_decoder.input_nc = 4 #type=int, default=4, help='# of input image channels')
opt_offset_decoder.output_nc = 4 #type=int, default=4, help='# of output image channels')
opt_offset_decoder.save_latest_freq = opt_offset_decoder.parent_exp.save_latest_freq
opt_offset_decoder.save_epoch_freq = opt_offset_decoder.parent_exp.save_epoch_freq
opt_offset_decoder.n_epochs = opt_offset_decoder.parent_exp.n_epochs
opt_offset_decoder.isTrain = opt_offset_decoder.parent_exp.isTrain #type=bool, default=True, help='whether to train the network or not')
opt_offset_decoder.continue_train = False #type=bool, default=False, help='continue training: load the latest model')
opt_offset_decoder.starting_epoch_count = opt_offset_decoder.parent_exp.starting_epoch_count #type=int, default=1, help='the starting epoch count, we save the model by <starting_epoch_count>, <starting_epoch_count>+<save_latest_freq>, ...')
# opt_offset_decoder.phase = opt_offset_decoder.parent_exp.phase #type=str, default='train', help='train, val, test, etc')
opt_offset_decoder.name = 'offset_decoder' #type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
opt_offset_decoder.loss_type = "L2_offset_loss" #type=string, default='L2_offset_loss', help='Loss type for the netowkr to enforce ['NoLoss'|'L1'|'L2'|'L1_sumL2'|'L2_sumL2'|'L2_sumL1'|'L2_offset_loss'|'L1_offset_loss'|'L1_sumL2_cross'|'L2_sumL2_cross']')
opt_offset_decoder.niter = 20 #type=int, default=100, help='# of iter at starting learning rate')
opt_offset_decoder.niter_decay = 100 #type=int, default=100, help='# of iter to linearly decay learning rate to zero')



opt_offset_decoder.gpu_ids = opt_offset_decoder.parent_exp.gpu_ids #type=str, default=['1','2','3','0'], help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
opt_offset_decoder.num_threads = 4 #default=4, type=int, help='# threads for loading data')
opt_offset_decoder.checkpoints_load_dir =  opt_offset_decoder.parent_exp.load_dir #type=str, default='./checkpoints', help='models are saved here')
opt_offset_decoder.checkpoints_save_dir =  opt_offset_decoder.parent_exp.checkpoints_dir #type=str, default='./checkpoints', help='models are saved here')
opt_offset_decoder.results_dir = opt_offset_decoder.parent_exp.results_dir
opt_offset_decoder.log_dir =  opt_offset_decoder.parent_exp.log_dir #type=str, default='./checkpoints', help='models are saved here')
opt_offset_decoder.verbose = False#type=bool, default=False, help='if specified, print more debugging information')
opt_offset_decoder.suffix ='' #default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
