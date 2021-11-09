#!/usr/bin/python
'''
Defines a generic wrapper class for all the network models
Utilizes params.py to create, initiate, load and train the network.
'''
import torch
from collections import OrderedDict
import scipy.io
from torch.autograd import Variable
import os
from Generators import *
from utils import *

class ModelADT():
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        
        self.opt = opt
#        self.gpu_ids = opt.gpu_ids
        gpu_ids = []
        for i in range(torch.cuda.device_count()):
            gpu_ids.append(str(i))
        print(gpu_ids)
        self.gpu_ids = gpu_ids
        self.isTrain = opt.isTrain
        self.loss_weight = opt.lambda_L
        self.reg_loss_weight = opt.lambda_reg
        self.cross_loss_weight = opt.lambda_cross
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) # if self.gpu_ids else torch.device('cpu')
        
        print(self.device)
        self.model_name = self.opt.name
        self.save_dir = os.path.join(self.opt.checkpoints_save_dir, self.model_name)
        self.load_dir = os.path.join(self.opt.checkpoints_load_dir, self.model_name)
        self.results_save_dir = opt.results_dir

        self.loss_names = []
        self.visual_names = []
        self.image_paths = []

        self.loss_names = ['loss_criterion']
        self.visual_names = ['output']
        

        self.net = get_model_funct(self.opt.net)(self.opt, self.gpu_ids)
        self.net = self.net.to(self.device)

        if self.isTrain:
            if self.opt.loss_type == "L2":
                self.loss_criterion = torch.nn.MSELoss()
            elif self.opt.loss_type == "L1":
                self.loss_criterion = torch.nn.L1Loss()
            elif self.opt.loss_type == "L1_sumL2":
                self.loss_criterion = torch.nn.L1Loss()
            elif self.opt.loss_type == "L2_sumL2":
                self.loss_criterion = torch.nn.MSELoss()
            elif self.opt.loss_type == "L2_sumL1":
                self.loss_criterion = torch.nn.MSELoss()
            elif self.opt.loss_type == "L2_offset_loss":
                self.loss_criterion = torch.nn.MSELoss()
            elif self.opt.loss_type == "L1_offset_loss":
                self.loss_criterion = torch.nn.L1Loss()
            elif self.opt.loss_type == "L1_sumL2_cross":
                self.loss_criterion = torch.nn.MSELoss()
            elif self.opt.loss_type == "L2_sumL2_cross":
                self.loss_criterion = torch.nn.MSELoss()
                self.cross_loss_criterion = torch.nn.NLLLoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer = torch.optim.Adam(self.net.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.starting_epoch_count)
        self.print_networks(opt.verbose)

    # make models eval mode during test time
    def eval(self):
        self.net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()
    
    def save_outputs(self):
        if not os.path.exists(self.results_save_dir):
            os.makedirs(self.results_save_dir, exist_ok=True)
        to_save_dict = {}
        for tensor in self.visual_names:
            tensor_val = getattr(self, tensor).data.cpu().numpy()
            to_save_dict[tensor] = tensor_val
        scipy.io.savemat(self.results_save_dir+".mat", mdict=to_save_dict)
            

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, epoch):
        name = self.model_name
        save_filename = '%s_net_%s.pth' % (epoch, name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir, save_filename)
        net = self.net
        print(save_path)
        print('net'+name)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available() and hasattr(net, 'module'):
            torch.save(net.module.cpu().state_dict(), save_path)
        else:
            torch.save(net.cpu().state_dict(), save_path)
        net.to(self.device)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, epoch, load_dir=""):
        """
        epoch (int/str): epoch index / "best" / "latest"
        """
        assert isinstance(epoch,int) or epoch=="best" or epoch=="latest"
        load_filename = f'{epoch}_net_{self.model_name}.pth'

        if load_dir:
            # use given load dir
            load_path = os.path.join(load_dir, self.model_name, load_filename) 
        else:
            # use default load dir
            load_path = os.path.join(self.load_dir, load_filename)

        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        print(state_dict.keys())
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)
        net = net.to(self.device)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        net = self.net
        name = self.model_name
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if verbose:
            print(net)
        print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # Set the input and target data for the network to train on/evaluate against
    def set_data(self, input, target, convert=False, shuffle_channel=False):
        shape_in = input.shape
        if shuffle_channel:
            self.input = input[:,torch.randperm(shape_in[1]),:,:]
        else:
            self.input = input
        self.target = target
        if convert:
            self.input, self.target = Variable(self.input), Variable(self.target)

        self.input = self.input.to(self.device)
        self.target = self.target.to(self.device)

    # Define the forward pass to compute loss
    def forward(self):
        self.output = self.net(self.input)
        if self.opt.loss_type != "NoLoss":
            self.loss = self.loss_weight*self.loss_criterion(self.output, self.target)

            if self.opt.loss_type == "L1_sumL2" or self.opt.loss_type == "L2_sumL2":
                self.loss += self.reg_loss_weight*torch.norm(self.output).div(self.output.numel())
                self.reg_loss = self.reg_loss_weight*torch.norm(self.output).div(self.output.numel())

            if self.opt.loss_type == "L2_sumL1":
                self.loss += self.reg_loss_weight*torch.norm(self.output,p=1).div(self.output.numel())
                self.reg_loss = self.reg_loss_weight*torch.norm(self.output,p=1).div(self.output.numel())

            if self.opt.loss_type == "L1_sumL2_cross" or self.opt.loss_type == "L2_sumL2_cross":
                self.loss += self.reg_loss_weight*torch.norm(self.output).div(self.output.numel())
                self.loss += self.cross_loss_weight*self.cross_loss_criterion(self.output.flatten(start_dim=1),self.target.flatten(start_dim=1))
            
            if self.opt.loss_type == "L1_offset_loss" or self.opt.loss_type == "L2_offset_loss":
                self.loss += self.reg_loss_weight*torch.norm(self.output).div(self.output.numel())

    def backward(self):
        self.loss.backward(retain_graph=True)
    
    def optimize_parameters(self):
        self.forward()
        self.backward()  
        self.optimizer.step()
        self.optimizer.zero_grad()
