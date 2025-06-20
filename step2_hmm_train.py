
import time
import logging

import torch
import configs
from models.data_loader import PolyphonicDataset
import models, configs
from models.helper import gVar
import pandas as pd

def save_model(model, epoch,train_loss,test_loss):
    ckpt_path='output/HMMmodel_epo{}.pkl'.format(epoch)
    print("saving model to %s..." % ckpt_path)
    torch.save(model.state_dict(), ckpt_path)
    pd.DataFrame({'train_loss': train_loss}).to_csv('output/HMMmodel_train_loss.csv.gz',compression='gzip')
    test_loss_values = [t.item() for t in test_loss]
    pd.DataFrame({'test_loss': test_loss_values}).to_csv('output/HMMmodel_test_loss.csv.gz',compression='gzip')


# setup, training, and evaluation
def train():
    
    # read config file
    config=getattr(configs, 'config_HMM')()
    
    
    # setup logging
    logging.basicConfig(filename='./output/HMMmodel.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(config)
        
    
    # instantiate the dmm
    input_dim = config['input_dim']
    z_dim = config['z_dim']
    model = models.HMM(input_dim,z_dim,config)
    model = model.cuda()
        
    train_set=PolyphonicDataset('data/sim/train.pkl')
    valid_set=PolyphonicDataset('data/sim/valid.pkl')
    test_set=PolyphonicDataset('data/sim/test.pkl')

    #################
    # TRAINING LOOP #
    #################
    
    times = [time.time()]
    train_loss = []
    test_loss = []
    for epoch in range(config['epochs']):
            
        train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True, num_workers=1)
        train_data_iter=iter(train_loader)
        n_iters=train_data_iter.__len__()
        
        epoch_nll = 0.0 # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
        i_batch=1   
        n_slices=0
        while True:            
            try: 
                x, x_rev, x_lens = next(train_data_iter)                  
            except StopIteration: break # end of epoch                 
            x, x_rev, x_lens = gVar(x), gVar(x_rev), gVar(x_lens)
            
            if config['anneal_epochs'] > 0 and epoch < config['anneal_epochs']: # compute the KL annealing factor            
                min_af = config['min_anneal']
                kl_anneal = min_af+(1.0-min_af)*(float(i_batch+epoch*n_iters+1)/float(config['anneal_epochs']*n_iters))
            else:            
                kl_anneal = 1.0 # by default the KL annealing factor is unity
            
            loss_AE = model.train_AE(x, x_lens, kl_anneal)
            
            epoch_nll += loss_AE['train_loss']
            i_batch=i_batch+1
            n_slices=n_slices+x_lens.sum().item()
            
        # do evaluation on test and validation data and report results
        train_l = epoch_nll/n_slices
        train_loss.append(train_l)

        if epoch % 10 == 0:
            
            times.append(time.time())
            epoch_time = times[-1] - times[-2]
            logging.info("[Epoch %04d]\t\t(dt = %.3f sec)"%(epoch, epoch_time))
            logging.info("[train epoch %08d]  %.8f" % (epoch, train_l))


            test_loader=torch.utils.data.DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=False, num_workers=1)
            for x, x_rev, x_lens in test_loader: 
                x, x_rev, x_lens = gVar(x), gVar(x_rev), gVar(x_lens)
                test_nll = model.valid_step(x, x_lens) / x_lens.sum()
                test_loss.append(test_nll)
            logging.info("[test epoch %08d]  %.8f" % (epoch, test_nll))
                 
    save_model(model,epoch,train_loss,test_loss)


train()