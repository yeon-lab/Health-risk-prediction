import numpy as np
import torch
from trainer import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
import pandas as pd
from model.utils import padding


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, 
                      optimizer,
                      metric_ftns,
                      config,
                      dataset):
        super().__init__(model, metric_ftns, optimizer, config)
        self.config = config

        self.batch_size = config["data_loader"]["args"]["batch_size"]
        self.apply_weight = True if config['hyper_params']['version'] == 'weight' else False
        model = config['hyper_params']['model'] 
        self.apply_DG = True if model == 'DG' or model == 'AdaDiag' else False
        self.out_dim = self.model.out_dim
        self.input_dim = config['hyper_params']['n_feat']
        
        self.train_x, self.train_y  = dataset['train']['DX'], dataset['train']['Y']
        self.train_df = pd.DataFrame(dataset['train'])
        self.n_samples = len(self.train_y)
        self.train_n_batches = int(np.ceil(float(self.n_samples) / float(self.batch_size)))

        self.valid_x, self.valid_y = dataset['valid']['DX'], dataset['valid']['Y']
        self.valid_df = pd.DataFrame(dataset['valid'])
        self.valid_n_batches = int(np.ceil(float(len(self.valid_y)) / float(self.batch_size)))
        
        self.test_x, self.test_y = dataset['test']['DX'], dataset['test']['Y']
        self.test_df = pd.DataFrame(dataset['test'])
        self.test_n_batches = int(np.ceil(float(len(self.test_y)) / float(self.batch_size)))
        
        if self.apply_weight:
            self.train_weights = dataset['weights']
            self.valid_weights = dataset['valid_weights']
        
        self.train_d, self.valid_d, self.test_d = None, None, None
        if self.apply_DG:
            self.train_d = dataset['train']['Domain']
            self.valid_d = dataset['valid']['Domain']
            self.test_d = dataset['test']['Domain']
            
        self.do_validation = dataset['valid'] is not None
        self.lr_scheduler = optimizer
        self.log_step = int(self.batch_size) * 1  # reduce this if you want more logs

        self.metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])


    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        self.metrics.reset()

        outs = torch.tensor([]).to(self.device)
        trgs = torch.tensor([]).to(dtype=torch.int64).to(self.device)
        for index in range(self.train_n_batches):
            self.optimizer.zero_grad()
            x = self.train_x[index*self.batch_size:(index+1)*self.batch_size]
            y = self.train_y[index*self.batch_size:(index+1)*self.batch_size]
            x, x_lengths = padding(x, self.input_dim)
            x_tensor = torch.from_numpy(x).float().to(self.device)
            y_tensor = torch.from_numpy(np.array(y)).float().to(self.device)
            
            if self.apply_weight:
                x_weights = self.train_weights[index*self.batch_size:(index+1)*self.batch_size]
                x_weights = torch.from_numpy(x_weights).float().to(self.device)
                pred, loss = self.model.predict(x_tensor, y_tensor, x_weights=x_weights)
            elif self.apply_DG:
                d = self.train_d[index*self.batch_size:(index+1)*self.batch_size]
                d_tensor = torch.from_numpy(np.array(d)).to(self.device)
                pred, loss = self.model.predict(x_tensor, y_tensor, d_tensor)
            else:
                pred, loss = self.model.predict(x_tensor, y_tensor)
            
            loss.backward()
            self.optimizer.step()
            self.metrics.update('loss', loss.item())


            if index % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(index),
                    loss.item(),
                ))
                
            outs = torch.cat([outs, pred])
            trgs = torch.cat([trgs, y_tensor])

        for met in self.metric_ftns:
            self.metrics.update(met.__name__, met(trgs, outs))
        log = self.metrics.result()

        if self.do_validation:
            val_log = self._infer(self.valid_x, self.valid_y, self.valid_d, self.valid_n_batches, do_validation = True)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        return log

        
    def _infer(self, x_data, y_data, d_data, n_batches, do_validation = False):
        self.model.eval()
        self.metrics.reset()
        with torch.no_grad():
            outs = torch.tensor([]).to(self.device)
            trgs = torch.tensor([]).to(dtype=torch.int64).to(self.device)
            for index in range(n_batches):
                x = x_data[index*self.batch_size:(index+1)*self.batch_size]
                y = y_data[index*self.batch_size:(index+1)*self.batch_size]
                x, x_lengths = padding(x, self.input_dim)
                x_tensor = torch.from_numpy(x).float().to(self.device)
                y_tensor = torch.from_numpy(np.array(y)).float().to(self.device)
                
                if do_validation and self.apply_weight:
                    x_weights = self.valid_weights[index*self.batch_size:(index+1)*self.batch_size]
                    x_weights = torch.from_numpy(x_weights).float().to(self.device)
                    pred, loss = self.model.predict(x_tensor, y_tensor, x_weights=x_weights)
                elif self.apply_DG:
                    d = d_data[index*self.batch_size:(index+1)*self.batch_size]
                    d_tensor = torch.from_numpy(np.array(d)).to(self.device)
                    pred, loss = self.model.predict(x_tensor, y_tensor, d_tensor)
                else:
                    pred, loss = self.model.predict(x_tensor, y_tensor)
                
                self.metrics.update('loss', loss.item())
                    
                outs = torch.cat([outs, pred])
                trgs = torch.cat([trgs, y_tensor])
                
        for met in self.metric_ftns:
            self.metrics.update(met.__name__, met(trgs, outs))
        
        return self.metrics.result()
        
    def _test_epoch(self):
        PATH = str(self.checkpoint_dir / 'model_best.pth')
        self.model.load_state_dict(torch.load(PATH)['state_dict'])
        self.model.eval()
        
        log = {}
        
        train_log = self._infer(self.train_x, self.train_y, self.train_d, self.train_n_batches)
        valid_log = self._infer(self.valid_x, self.valid_y, self.valid_d, self.valid_n_batches)
        test_log = self._infer(self.test_x, self.test_y, self.test_d, self.test_n_batches)
        log.update(**{'train_' + k: v for k, v in train_log.items()})
        log.update(**{'val_' + k: v for k, v in valid_log.items()})
        log.update(**{'test_' + k: v for k, v in test_log.items()})
                
        self.logger.info('='*100)
        self.logger.info('Inference is completed')
        self.logger.info('-'*100)
        for key, value in log.items():
            self.logger.info('    {:20s}: {}'.format(str(key), value))  
        self.logger.info('='*100)

        return log 
        
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx * self.batch_size
        total = self.n_samples 

        return base.format(current, total, 100.0 * current / total)
        
        
