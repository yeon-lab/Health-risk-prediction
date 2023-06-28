import argparse
import collections
import numpy as np
import pandas as pd
import os
import random
import torch
import pickle

import model.loss as module_loss
import model.models as model_arch
import model.metric as module_metric
from utils.parse_config import ConfigParser
from trainer import Trainer
from model.sample_weighting import *
from utils.load_data import init_data

def main(params, config, dataset, version):

    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    model = getattr(model_arch, params.model)
    model = model(config, criterion)
    model.weights_init()  

    logger = config.get_logger('train') 

    # build optimizer
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, model_parameters)
    
    if version == 'weight':
        dataset = reweight_data(dataset, config['hyper_params']["max_visit"], config['hyper_params']['n_feat'], params.steps, params.step_lr, params.kl_weight, params.dist_weight, params.kl_dim)
    trainer = Trainer(model, 
                      optimizer,
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
    args.add_argument('--version', default='basic', type=str)
    args.add_argument('--time', type=int)
    args.add_argument('--target', type=str)
    args.add_argument('--day_dim', type=int)
    args.add_argument('--rnn_hidden', type=int)
    args.add_argument('--model', type=str)
    args.add_argument('--weight_decay', type=float)
    args.add_argument('--step_lr', type=float)
    args.add_argument('--steps', type=int)
    args.add_argument('--kl_weight', type=float)
    args.add_argument('--dist_weight', type=float)
    args.add_argument('--kl_dim', type=int)
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    params = args.parse_args()
    
    data_file = 'pickles/input_{}_{}.pkl'.format(params.target, params.time)   
    
    version = params.version
    if params.model == 'AdaDiag' or params.model == 'DG':
        version = 'basic'
    exper_name = '{}_{}'.format(params.target, params.time)
    config = ConfigParser.from_args(args, exper_name)
    config['hyper_params']['model'] = params.model
    config['hyper_params']['day_dim']= params.day_dim
    config['hyper_params']['rnn_hidden']= params.rnn_hidden
    config['hyper_params']['version'] = version
    config['optimizer']['args']['weight_decay']= params.weight_decay
    config['hyper_params']["min_visit"], config['hyper_params']["max_visit"] = 10, 30

    dataset, config, feat_dict, y_dict = init_data(data_file, config)
    config['hyper_params']['n_feat'], config['hyper_params']['n_class'] = len(feat_dict.keys()), len(y_dict.keys())

    log = main(params, config, dataset, version) 


        
