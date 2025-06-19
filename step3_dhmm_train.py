"""
An implementation of a Deep Markov Model in Pyro based on reference [1].

Adopted from https://github.com/uber/pyro/tree/dev/examples/dmm  
         and https://github.com/clinicalml/structuredinference

Reference:

[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""

import time
import logging

import torch
import configs
from models.data_loader import PolyphonicDataset
import models, configs
from models.helper import gVar

def save_model(model, epoch):
    ckpt_path='output/model_epo{}.pkl'.format('DHMM', epoch)
    print("saving model to %s..." % ckpt_path)
    torch.save(model.state_dict(), ckpt_path)

# def load_model(model, epoch):
#     ckpt_path='./output/{}/{}/{}/models/model_epo{}.pkl'.format(args.model, args.expname, args.dataset, epoch)
#     assert exists(ckpt_path), "epoch misspecified"
#     print("loading model from %s..." % ckpt_path)
#     model.load_state_dict(torch.load(ckpt_path))


# setup, training, and evaluation
def train():
    
    # read config file
    config=getattr(configs, 'config_DHMM')()
    
    
    # setup logging
    logging.basicConfig(filename='./output/model.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(config)
        
    
    # instantiate the dmm
    model = getattr(models, 'DHMM')(config)
    model = model.cuda()
        
    train_set=PolyphonicDataset('data/polyphonic/train.pkl')
    valid_set=PolyphonicDataset('data/polyphonic/valid.pkl')
    test_set=PolyphonicDataset('data/polyphonic/test.pkl')

    #################
    # TRAINING LOOP #
    #################
    
    times = [time.time()]
    for epoch in range(config['epochs']):
            
        train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True, num_workers=1)
        train_data_iter=iter(train_loader)
        n_iters=train_data_iter.__len__()
        
        epoch_nll = 0.0 # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
        i_batch=1   
        n_slices=0
        loss_records={}
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
            
            loss_AE = model.train_AE(x, x_rev, x_lens, kl_anneal)
            
            epoch_nll += loss_AE['train_loss_AE']
            i_batch=i_batch+1
            n_slices=n_slices+x_lens.sum().item()
            
        loss_records.update(loss_AE)   
        loss_records.update({'epo_nll':epoch_nll/n_slices})
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        logging.info("[Epoch %04d]\t\t(dt = %.3f sec)"%(epoch, epoch_time))
        logging.info(loss_records)

        # do evaluation on test and validation data and report results
        if (epoch) % 10 == 0:
            test_loader=torch.utils.data.DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=False, num_workers=1)
            for x, x_rev, x_lens in test_loader: 
                x, x_rev, x_lens = gVar(x), gVar(x_rev), gVar(x_lens)
                test_nll = model.valid(x, x_rev, x_lens) / x_lens.sum()
            logging.info("[val/test epoch %08d]  %.8f" % (epoch, test_nll))
    save_model(model, epoch)


train()