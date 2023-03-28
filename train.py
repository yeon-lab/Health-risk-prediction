import argparse
import collections
import numpy as np

import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *
from model import *

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
                    
def main(params, config, dataset):
    if config['arch']['type'] == 'Retain':
        model = Retain(config=config)
    elif config['arch']['type'] == 'Dipole':
        model = Dipole(config=config)
    elif config['arch']['type'] == 'LSTM':
        model = LSTM(config=config)
    elif config['arch']['type'] == 'GRU':
        model = GRU(config=config)
    elif config['arch']['type'] == 'Concare':
        model = Concare(config=config)
    elif config['arch']['type'] == 'Stagenet':
        model = Stagenet(config=config)

    model.apply(weights_init_normal)
    logger.info(model)
    logger.info("-"*100)


    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, model_parameters)
    
    if params.version == 'weight':
        dataset = reweight_data(dataset, config['hyper_params']["max_visit"], config['hyper_params']['n_feat'], params.steps, params.step_lr, params.kl_weight, 
                                    params.dist_weight, params.kl_dim)

    trainer = Trainer(model, 
                      optimizer,
                      criterion, 
                      metrics,
                      config=config,
                      dataset=dataset
                      )

    return trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', type=str)
    args.add_argument('--version', type=str)
    args.add_argument('--day_dim', type=int)
    args.add_argument('--rnn_hidden', type=int)
    args.add_argument('--model', type=str)
    args.add_argument('--weight_decay', type=float)
    args.add_argument('--step_lr', type=float)
    args.add_argument('--steps', type=int)
    args.add_argument('--kl_weight', type=float)
    args.add_argument('--dist_weight', type=float)
    args.add_argument('--kl_dim', type=int)
    args.add_argument('--np_data_dir', type=str,
                      help='Directory containing numpy files')
    args.add_argument('--data_file',type=str)

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    params = args.parse_args()
    config = ConfigParser.from_args(args)
    config['hyper_params']['day_dim']= params.day_dim
    config['hyper_params']['rnn_hidden']= params.rnn_hidden       
    
    dataset, feat_dict, y_dict = init_data(data_file = args2.data_file, npy_path=args2.np_data_dir, config=config)
    config['hyper_params']['n_feat'], config['hyper_params']['n_class'] = len(feat_dict.keys()), len(y_dict.keys())

    log, log_per_month = main(params, config, dataset)
    
    np.save('{}_{}_log_per_month'.format(params.version, config['arch']['type']), log_per_month) 
    np.save('{}_{}_log'.format(params.version, config['arch']['type']), log) 
