import argparse
import collections
import numpy as np

import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *
from model.Dipole import *

import torch
import torch.nn as nn


# fix random seeds for reproducibility
SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def weights_init_normal(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight.data, -0.1, 0.1)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif type(m) == nn.GRU:
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.uniform_(m.__getattr__(p), -0.1, 0.1)
                if 'bias' in p:
                    torch.nn.init.constant_(m.__getattr__(p), 0.0)
                    
def main(config, dataset):
    logger = config.get_logger('train') 
    logger.info('='*100)
    for key, value in config['hyper_params'].items():
        logger.info('    {:25s}: {}'.format(str(key), value))
    for key, value in config['data_loader']['args'].items():
        logger.info('    {:25s}: {}'.format(str(key), value))
    logger.info('    {:25s}: {}'.format('optimizer', config['optimizer']['type']))
    for key, value in config['optimizer']['args'].items():
        logger.info('    {:25s}: {}'.format(str(key), value))
    for key, value in config['trainer'].items():
        logger.info('    {:25s}: {}'.format(str(key), value))

    
    logger.info("="*100)
        
    
    # build model architecture, initialize weights, then print to console  
    
    model = Dipole(config=config)
    model.apply(weights_init_normal)

    logger.info(model)
    logger.info("-"*100)


    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, model_parameters)

    trainer = Trainer(model, 
                      optimizer,
                      criterion, 
                      metrics,
                      config=config,
                      train_data=dataset['train'],
                      valid_data=dataset['valid'],
                      test_data_ICD9=dataset['test_ICD9'],
                      test_data_ICD10=dataset['test_ICD10']
                      )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--np_data_dir', default = './data_npy/',  type=str,
                      help='Directory containing numpy files')
    args.add_argument('--data_file', default = 'input_outpat.npy',type=str)
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    args2 = args.parse_args()
    config = ConfigParser.from_args(args)
       
    dataset, feat_dict, y_dict = init_data(data_file = args2.data_file, npy_path=args2.np_data_dir, config=config)
    config['hyper_params']['n_feat'], config['hyper_params']['n_class'] = len(feat_dict.keys()), len(y_dict.keys())

    main(config, dataset)
