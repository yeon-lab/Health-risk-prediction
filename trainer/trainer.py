import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
import pandas as pd

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, 
                      optimizer,
                      criterion, 
                      metric_ftns,
                      config,
                      dataset):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config

        self.batch_size = config["data_loader"]["args"]["batch_size"]
        self.apply_weight = True if config['hyper_params']['version'] == 'weight' else False
        self.out_dim = self.model.out_dim
        
        self.train_x = dataset['train']['DX'].tolist()
        self.train_y = dataset['train']['Y'].tolist()
        self.train_df = pd.DataFrame(dataset['train'])
        self.n_samples = len(self.train_y)
        self.train_n_batches = int(np.ceil(float(self.n_samples) / float(self.batch_size)))

        self.valid_x = dataset['valid']['DX'].tolist()
        self.valid_y = dataset['valid']['Y'].tolist()
        self.valid_df = pd.DataFrame(dataset['valid'])
        self.valid_n_batches = int(np.ceil(float(len(self.valid_y)) / float(self.batch_size)))
        
        self.test_x = dataset['test']['DX'].tolist()
        self.test_y = dataset['test']['Y'].tolist()
        self.test_df = pd.DataFrame(dataset['test'])
        self.test_n_batches = int(np.ceil(float(len(self.test_y)) / float(self.batch_size)))
        
        if self.apply_weight:
            self.train_weights = dataset['weights']
            self.valid_weights = dataset['valid_weights']
            
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
            x, x_lengths = self.model.padMatrixWithoutTime(seqs=x)
            x_tensor = torch.from_numpy(x).float().to(self.device)

            if self.out_dim == 1:
                y_tensor = torch.from_numpy(np.array(y)).float().to(self.device)
            else:
                y_tensor = torch.from_numpy(np.array(y)).long().to(self.device)
            
            pred = self.model(x_tensor)
            pred = pred.squeeze(1)
            
            if self.apply_weight:
                x_weights = self.train_weights[index*self.batch_size:(index+1)*self.batch_size]
                x_weights = torch.from_numpy(x_weights).float().to(self.device)
                losses = self.criterion(pred, y_tensor).view(1, -1)
                w_weights = x_weights.view(-1, 1)
                weighted_loss = losses.mm(w_weights)
                loss = weighted_loss.view(1)
            else:
                loss = self.criterion(pred, y_tensor).mean()
            

            loss.backward()
            self.optimizer.step()
            self.metrics.update('loss', loss.item())
            
            if epoch == 1 and index==0:
                print('x.shape:', x.shape)
                print('x_tensor.shape:', x_tensor.shape)


            if index % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(index),
                    loss.item(),
                ))
                
            outs = torch.cat([outs, pred])
            trgs = torch.cat([trgs, y_tensor])

            
        for met in self.metric_ftns:
            self.metrics.update(met.__name__, met(trgs, outs, self.out_dim))
        log = self.metrics.result()

        if self.do_validation:
            val_log = self._infer(self.valid_x, self.valid_y, self.valid_n_batches, do_validation = True)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        return log

        
    def _infer(self, x_data, y_data, n_batches, do_validation = False):
        self.model.eval()
        self.metrics.reset()
        with torch.no_grad():
            outs = torch.tensor([]).to(self.device)
            trgs = torch.tensor([]).to(dtype=torch.int64).to(self.device)
            for index in range(n_batches):
                x = x_data[index*self.batch_size:(index+1)*self.batch_size]
                y = y_data[index*self.batch_size:(index+1)*self.batch_size]
                x, x_lengths = self.model.padMatrixWithoutTime(seqs=x)
                x_tensor = torch.from_numpy(x).float().to(self.device)
                if self.out_dim == 1:
                    y_tensor = torch.from_numpy(np.array(y)).float().to(self.device)
                else:
                    y_tensor = torch.from_numpy(np.array(y)).long().to(self.device)
                    
                    
                pred = self.model(x_tensor)
                pred = pred.squeeze(1)
                
                if do_validation and self.apply_weight:
                    x_weights = self.valid_weights[index*self.batch_size:(index+1)*self.batch_size]
                    x_weights = torch.from_numpy(x_weights).float().to(self.device)
                    losses = self.criterion(pred, y_tensor).view(1, -1)
                    w_weights = x_weights.view(-1, 1)
                    weighted_loss = losses.mm(w_weights)
                    loss = weighted_loss.view(1)
                else:
                    loss = self.criterion(pred, y_tensor).mean()
                
                self.metrics.update('loss', loss.item())
                    
                outs = torch.cat([outs, pred])
                trgs = torch.cat([trgs, y_tensor])
                
        for met in self.metric_ftns:
            self.metrics.update(met.__name__, met(trgs, outs, self.out_dim))
        
        return self.metrics.result()
        
    def _infer_per_time(self, df, time='M'):
        self.model.eval()
        log = {}
        with torch.no_grad():        
            for time_step, group in df.set_index('last_observed_date').groupby(pd.Grouper(freq=time)):
                self.metrics.reset()
                if len(group) == 0:
                    continue
                x = group.DX.tolist()
                y = group.Y.tolist()
                x, x_lengths = self.model.padMatrixWithoutTime(seqs=x)
                x_tensor = torch.from_numpy(x).float().to(self.device)
                if self.out_dim == 1:
                    y_tensor = torch.from_numpy(np.array(y)).float().to(self.device)
                else:
                    y_tensor = torch.from_numpy(np.array(y)).long().to(self.device)
                pred = self.model(x_tensor)
                pred = pred.squeeze(1)
                loss = self.criterion(pred, y_tensor).mean()
                
                self.metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.metrics.update(met.__name__, met(y_tensor, pred, self.out_dim))
                    
                if time == 'M':
                    date = "{}-{}".format(time_step.strftime("%Y"), time_step.strftime("%m"))
                elif time == 'Y':
                    date = "{}".format(time_step.strftime("%Y"))
                    
                log[date] = {}
                log[date].update({'patients': len(group)})   
                log[date].update(**{k: v for k, v in self.metrics.result().items()})   
        
        return log
        
    def _test_epoch(self):
        PATH = str(self.checkpoint_dir / 'model_best.pth')
        self.model.load_state_dict(torch.load(PATH)['state_dict'])
        self.model.eval()
        
        log = {}
        log_per_month = {}
        log_per_year = {}
        for phase in ['train', 'valid', 'test']:
            log_per_month[phase] = {}
            log_per_year[phase] = {}
        
        
        train_log = self._infer(self.train_x, self.train_y, self.train_n_batches)
        valid_log = self._infer(self.valid_x, self.valid_y, self.valid_n_batches)
        test_log = self._infer(self.test_x, self.test_y, self.test_n_batches)
        log.update(**{'train_' + k: v for k, v in train_log.items()})
        log.update(**{'val_' + k: v for k, v in valid_log.items()})
        log.update(**{'test_' + k: v for k, v in test_log.items()})
        
        log_per_month['train'].update(**{k: v for k, v in self._infer_per_time(self.train_df, time='M').items()})
        log_per_month['valid'].update(**{k: v for k, v in self._infer_per_time(self.valid_df, time='M').items()})
        log_per_month['test'].update(**{k: v for k, v in self._infer_per_time(self.test_df, time='M').items()})

        
                
        self.logger.info('='*100)
        self.logger.info('Inference is completed')
        self.logger.info('-'*100)
        for key, value in log.items():
            self.logger.info('    {:20s}: {}'.format(str(key), value))  
        self.logger.info('='*100)

            
        return log, log_per_month #, log_per_year
        
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx * self.batch_size
        total = self.n_samples 

        return base.format(current, total, 100.0 * current / total)
        
