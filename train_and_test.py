#!/usr/bin/python
'''
Script for both training and evaluating the DLoc network
Automatically imports the parameters from params.py.
For further details onto which params file to load
read the README in `params_storage` folder.
'''

import torch
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
from utils import *
from modelADT import ModelADT
from Generators import *
from data_loader import load_data
from joint_model import Enc_2Dec_Network
from joint_model import Enc_Dec_Network
from params import *
import trainer
torch.manual_seed(0)
np.random.seed(0)

'''
Defining the paths from where to Load Data.
Assumes that the data is stored in a subfolder called data in the current data folder
'''

#####################################Final Simple Space Results################################################
if "data" in opt_exp and opt_exp.data == "rw_to_rw_atk":
    # Training and testing data loaded for the Final results For Env-1 (The smaller space) in the paper (Figure 10a)
    trainpath = ['./data/dataset_non_fov_train_July18.mat',
                './data/dataset_fov_train_July18.mat']
    testpath = ['./data/dataset_non_fov_test_July18.mat',
                './data/dataset_fov_test_July18.mat']
    print('Real World to Real World experiments started')

#####################################Final Complex Space Results################################################
elif "data" in opt_exp and opt_exp.data == "rw_to_rw":
    # Training and testing data loaded for the Final results For Env-2 (The larger space) in the paper (Figure 10b)
    trainpath = ['./data/dataset_edit_jacobs_July28.mat',
                './data/dataset_non_fov_train_jacobs_July28_2.mat',
                './data/dataset_fov_train_jacobs_July28_2.mat']
    testpath = ['./data/dataset_fov_test_jacobs_July28_2.mat',
                './data/dataset_non_fov_test_jacobs_July28_2.mat']
    print('Real World to Real World experiments started')

#########################################Generalization across Scenarios###########################################

elif "data" in opt_exp and opt_exp.data == "rw_to_rw_env2":
    # Training and testing data loaded for the Final results For Env-2
    # for Generalization across scenarios (Table-1) train on 1/3/4 and test on 2
    trainpath = ['./data/dataset_edit_jacobs_July28.mat',
                './data/dataset_non_fov_train_jacobs_July28_2.mat',
                './data/dataset_fov_train_jacobs_July28_2.mat',
                './data/dataset_train_jacobs_Aug16_3.mat',
                './data/dataset_train_jacobs_Aug16_4_ref.mat']
    testpath = ['./data/dataset_train_jacobs_Aug16_1.mat']
    print('Real World to Real World experiments started')


elif "data" in opt_exp and opt_exp.data == "rw_to_rw_env3":
    # Training and testing data loaded for the Final results For Env-2
    # for Generalization across scenarios (Table-1) train on 1/2/4 and test on 3
    trainpath = ['./data/dataset_edit_jacobs_July28.mat',
                './data/dataset_non_fov_train_jacobs_July28_2.mat',
                './data/dataset_fov_train_jacobs_July28_2.mat',
                './data/dataset_train_jacobs_Aug16_1.mat',
                './data/dataset_train_jacobs_Aug16_4_ref.mat']
    testpath = ['./data/dataset_train_jacobs_Aug16_3.mat']
    print('Real World to Real World experiments started')

elif "data" in opt_exp and opt_exp.data == "rw_to_rw_env4":
    # Training and testing data loaded for the Final results For Env-2
    # for Generalization across scenarios (Table-1) train on 1/2/3 and test on 4
    trainpath = ['./data/dataset_edit_jacobs_July28.mat',
                './data/dataset_non_fov_train_jacobs_July28_2.mat',
                './data/dataset_fov_train_jacobs_July28_2.mat',
                './data/dataset_train_jacobs_Aug16_1.mat',
                './data/dataset_train_jacobs_Aug16_3.mat']
    testpath = ['./data/dataset_train_jacobs_Aug16_4_ref.mat']
    print('Real World to Real World experiments started')

######################################Generalization Across Bandwidth##########################################

elif "data" in opt_exp and opt_exp.data == "rw_to_rw_40":
    # Training and testing data loaded for the Generalization results For Env-2 (The larger space) in the paper (Figure 13a) at 40MHz
    trainpath = ['./data/dataset40_edit_jacobs_July28.mat',
                './data/dataset40_non_fov_train_jacobs_July28_2.mat',
                './data/dataset40_fov_train_jacobs_July28_2.mat']
    testpath = ['./data/dataset40_fov_test_jacobs_July28_2.mat',
                './data/dataset40_non_fov_test_jacobs_July28_2.mat']
    print('Real World to Real World experiments started')

elif "data" in opt_exp and opt_exp.data == "rw_to_rw_20":
    # Training and testing data loaded for the Generalization results For Env-2 (The larger space) in the paper (Figure 13a) at 20MHz
    trainpath = ['./data/dataset20_edit_jacobs_July28.mat',
                './data/dataset20_non_fov_train_jacobs_July28_2.mat',
                './data/dataset20_fov_train_jacobs_July28_2.mat']
    testpath = ['./data/dataset20_fov_test_jacobs_July28_2.mat',
                './data/dataset20_non_fov_test_jacobs_July28_2.mat']
    print('Real World to Real World experiments started')

######################################Generalization Across Space##########################################

elif "data" in opt_exp and opt_exp.data == "data_segment":
    # Training and testing data loaded for the Final results For Env-2 
    # for Disjoint Training and Testing(The larger space) in the paper (Figure 13b)
    trainpath = ['./data/dataset_test_jacobs_July28.mat',
                './data/dataset_test_jacobs_July28_2.mat']
    testpath = ['./data/dataset_train_jacobs_July28.mat',
                './data/dataset_train_jacobs_July28_2.mat']
    print('non-FOV to non-FOV experiments started')

######################################################################################################################
'''
Loading Training and Evaluation Data into their respective Dataloaders
'''
# load traning data
B_train,A_train,labels_train = load_data(trainpath[0], 0, 0, 0, 1)

for i in range(len(trainpath)-1):
    f,f1,l = load_data(trainpath[i+1], 0, 0, 0, 1)
    B_train = torch.cat((B_train, f), 0)
    A_train = torch.cat((A_train, f1), 0)
    labels_train = torch.cat((labels_train, l), 0)

labels_train = torch.unsqueeze(labels_train, 1)

train_data = torch.utils.data.TensorDataset(B_train, A_train, labels_train)
train_loader =torch.utils.data.DataLoader(train_data, batch_size=opt_exp.batch_size, shuffle=True)

print(f"A_train.shape: {A_train.shape}")
print(f"B_train.shape: {B_train.shape}")
print(f"labels_train.shape: {labels_train.shape}")
print('# training mini batch = %d' % len(train_loader))

# load testing data
B_test,A_test,labels_test = load_data(testpath[0], 0, 0, 0, 1)

for i in range(len(testpath)-1):
    f,f1,l = load_data(testpath[i+1], 0, 0, 0, 1)
    B_test = torch.cat((B_test, f), 0)
    A_test = torch.cat((A_test, f1), 0)
    labels_test = torch.cat((labels_test, l), 0)

labels_test = torch.unsqueeze(labels_test, 1)

# create data loader
test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)
test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_exp.batch_size, shuffle=False)
print(f"A_test.shape: {A_test.shape}")
print(f"B_test.shape: {B_test.shape}")
print(f"labels_test.shape: {labels_test.shape}")
print('# testing mini batch = %d' % len(test_loader))
print('Test Data Loaded')

'''
Initiate the Network and build the graph
'''

# init encoder
enc_model = ModelADT()
enc_model.initialize(opt_encoder)
enc_model.setup(opt_encoder)

# init decoder1
dec_model = ModelADT()
dec_model.initialize(opt_decoder)
dec_model.setup(opt_decoder)

if opt_exp.n_decoders == 2:
    # init decoder2
    offset_dec_model = ModelADT()
    offset_dec_model.initialize(opt_offset_decoder)
    offset_dec_model.setup(opt_offset_decoder)

    # join all models
    print('Making the joint_model')
    joint_model = Enc_2Dec_Network()
    joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, gpu_ids=opt_exp.gpu_ids)

elif opt_exp.n_decoders == 1:
    # join all models
    print('Making the joint_model')
    joint_model = Enc_Dec_Network()
    joint_model.initialize(opt_exp, enc_model, dec_model, gpu_ids=opt_exp.gpu_ids)

else:
    print('Incorrect number of Decoders specified in the parameters')
    return -1

if opt_exp.isFrozen:
    enc_model.load_networks(opt_encoder.starting_epoch_count)
    dec_model.load_networks(opt_decoder.starting_epoch_count)
    if opt_exp.n_decoders == 2:
        offset_dec_model.load_networks(opt_offset_decoder.starting_epoch_count)

# train the model
'''
Trainig the network
'''
trainer.train(joint_model, train_loader, test_loader)

'''
Model Evaluation at the best epoch
'''

epoch = "best"  # int/"best"/"last"
# load network
enc_model.load_networks(epoch, load_dir=eval_name)
dec_model.load_networks(epoch, load_dir=eval_name)
if opt_exp.n_decoders == 2:
    offset_dec_model.load_networks(epoch, load_dir=eval_name)
    joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, gpu_ids = opt_exp.gpu_ids)
elif opt_exp.n_decoders == 1:
    joint_model.initialize(opt_exp, enc_model, dec_model, gpu_ids = opt_exp.gpu_ids)

# pass data through model
total_loss, median_error = trainer.test(joint_model, 
    test_loader, 
    save_output=True,
    save_dir=eval_name,
    save_name=f"decoder_test_result_epoch_{epoch}",
    log=False)
print(f"total_loss: {total_loss}, median_error: {median_error}")
