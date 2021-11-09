'''
Scripts for the training and testing functions
train() function is called for training the network
test() function is called to evaluate the network
Both the function logs and saves the results in the files 
as mentioned in the params.py file
'''
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
import numpy as np  
import os
import time
import hdf5storage
from utils import *
from Generators import *
from params import *

def train(model, train_loader, test_loader):
    """Traning pipeline

    Args:
        model (torch.module): pytorch model
        train_loader (torch.dataloader): dataloader
        test_loader (torch.dataloader): dataloader
    """
    # set data index
    offset_output_index=0
    input_index=1
    output_index=2
    
    # initialization
    total_steps = 0
    print('Training called')
    stopping_count = 0

    for epoch in range(model.opt.starting_epoch_count+1, model.opt.n_epochs+1): # opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_offset_loss = 0
        error =[]

        for i, data in enumerate(train_loader):
            total_steps += model.opt.batch_size
            if opt_exp.n_decoders == 2:
                model.set_input(data[input_index], data[output_index], data[offset_output_index], shuffle_channel=False)
            elif opt_exp.n_decoders == 1:
                model.set_input(data[input_index], data[output_index], shuffle_channel=False)
            model.optimize_parameters()
            dec_outputs = model.decoder.output
            # print(f"dec_outputs size is : {dec_outputs.shape}")
            error.extend(localization_error(dec_outputs.data.cpu().numpy(),data[output_index].cpu().numpy(),scale=0.1))

            write_log([str(model.decoder.loss.item())], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='loss')
            if opt_exp.n_decoders == 2:
                write_log([str(model.offset_decoder.loss.item())], model.offset_decoder.model_name, log_dir=model.offset_decoder.opt.log_dir, log_type='offset_loss')
            if total_steps % model.decoder.opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            epoch_loss += model.decoder.loss.item()
            if opt_exp.n_decoders == 2:
                epoch_offset_loss += model.offset_decoder.loss.item()

        median_error_tr = np.median(error)
        error_90th_tr = np.percentile(error,90)
        error_99th_tr = np.percentile(error,99)
        nighty_percentile_error_tr = np.percentile(error,90)
        epoch_loss /= i
        if opt_exp.n_decoders == 2:
            epoch_offset_loss /= i
        write_log([str(epoch_loss)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='epoch_decoder_loss')
        if opt_exp.n_decoders == 2:
            write_log([str(epoch_offset_loss)], model.offset_decoder.model_name, log_dir=model.offset_decoder.opt.log_dir, log_type='epoch_offset_decoder_loss')
        write_log([str(median_error_tr)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='train_median_error')
        write_log([str(error_90th_tr)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='train_90th_error')
        write_log([str(error_99th_tr)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='train_99th_error')
        write_log([str(nighty_percentile_error_tr)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='train_90_error')
        if (epoch==1):
            min_eval_loss, median_error = test(model, test_loader, save_output=False)
        else:
            new_eval_loss, new_med_error = test(model, test_loader, save_output=False)
            if (median_error>=new_med_error):
                stopping_count = stopping_count+1
                median_error = new_med_error

        # generated_outputs = temp_generator_outputs
        if epoch % model.encoder.opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
            if (stopping_count==2):
                print('Saving best model at %d epoch' %(epoch))
                model.save_networks('best')
                stopping_count=0

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, model.decoder.opt.niter + model.decoder.opt.niter_decay, time.time() - epoch_start_time))
        model.decoder.update_learning_rate()
        model.encoder.update_learning_rate()
        if opt_exp.n_decoders == 2:
            model.offset_decoder.update_learning_rate()


def test(model, test_loader, save_output=True, save_name="decoder_test_result", save_dir="", log=True):
    """Test and evaluation pipeline

    Args:
        model (torch.module): pytorch model
        test_loader (torch.dataloader): dataloader
        save_output (bool, optional): whether to save output to mat file. Defaults to True.
        save_name (str, optional): name of the mat file. Defaults to "decoder_test_result".
        save_dir (str, optional): directory where output mat file is saved. Defaults to "".
        log (bool, optional): whether to log output. Defaults to True.

    Returns:
        tuple: (total_loss -> float, median_error -> float)
    """
    print('Evaluation Called')
    model.eval()

    # set data index
    offset_output_index=0
    input_index=1
    output_index=2

    # create containers
    generated_outputs = []
    offset_outputs = []
    total_loss = 0
    total_offset_loss = 0
    error =[]
    for i, data in enumerate(test_loader):
        if opt_exp.n_decoders == 2:
                model.set_input(data[input_index], data[output_index], data[offset_output_index], shuffle_channel=False)
            elif opt_exp.n_decoders == 1:
                model.set_input(data[input_index], data[output_index], shuffle_channel=False)
        model.test()

        # get model outputs
        gen_outputs = model.decoder.output  # gen_outputs.size = (N,1,H,W)
        if opt_exp.n_decoders == 2:
            off_outputs = model.offset_decoder.output # off_outputs.size = (N,n_ap,H,W)

        generated_outputs.extend(gen_outputs.data.cpu().numpy())
        if opt_exp.n_decoders == 2:
            offset_outputs.extend(off_outputs.data.cpu().numpy())
        error.extend(localization_error(gen_outputs.data.cpu().numpy(),data[output_index].cpu().numpy(),scale=0.1))
        total_loss += model.decoder.loss.item()
        if opt_exp.n_decoders == 2:
            total_offset_loss += model.offset_decoder.loss.item()
    total_loss /= i
    if opt_exp.n_decoders == 2:
        total_offset_loss /= i
    median_error = np.median(error)
    nighty_percentile_error = np.percentile(error,90)
    error_99th = np.percentile(error,99)

    if log:
        write_log([str(median_error)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_median_error')
        write_log([str(nighty_percentile_error)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_90_error')
        write_log([str(error_99th)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_99_error')
        write_log([str(total_loss)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_loss')
        if opt_exp.n_decoders == 2:
            write_log([str(total_offset_loss)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_offset_loss')

    if save_output:
        if not save_dir:
            save_dir = model.decoder.results_save_dir # default save directory

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        save_path = f"{save_dir}/{save_name}.mat"
        hdf5storage.savemat(save_path,
            mdict={"outputs":generated_outputs,"wo_outputs":offset_outputs, "error": error}, 
            appendmat=True, 
            format='7.3',
            truncate_existing=True)
        print(f"result saved in {save_path}")
    return total_loss, median_error