import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, 
                      optimizer,
                      criterion, 
                      metric_ftns,
                      config,
                      train_data,
                      valid_data,
                      test_data_ICD9,
                      test_data_ICD10):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.batch_size = config["data_loader"]["args"]["batch_size"]
        self.n_class = config['hyper_params']['n_class']
        
        self.train_x = train_data['X'].tolist()
        self.train_y = train_data['Y'].tolist()
        self.n_samples = len(self.train_y)
        self.train_n_batches = int(np.ceil(float(self.n_samples) / float(self.batch_size)))

        self.valid_x = valid_data['X'].tolist()
        self.valid_y = valid_data['Y'].tolist()
        self.valid_n_batches = int(np.ceil(float(len(self.valid_y)) / float(self.batch_size)))
        
        self.test_ICD9_x = test_data_ICD9['X'].tolist()
        self.test_ICD9_y = test_data_ICD9['Y'].tolist()
        self.test_ICD9_n_batches = int(np.ceil(float(len(self.test_ICD9_y)) / float(self.batch_size)))
        
        self.test_ICD10_x = test_data_ICD10['X'].tolist()
        self.test_ICD10_y = test_data_ICD10['Y'].tolist()
        self.test_ICD10_n_batches = int(np.ceil(float(len(self.test_ICD10_y)) / float(self.batch_size)))

        self.do_validation = valid_data is not None
        self.lr_scheduler = optimizer
        self.log_step = int(self.batch_size) * 1  # reduce this if you want more logs

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.test_ICD9_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.test_ICD10_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])


    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        outs = torch.tensor([]).to(self.device)
        trgs = torch.tensor([]).to(dtype=torch.int64).to(self.device)
        for index in range(self.train_n_batches):
            self.optimizer.zero_grad()
            x = self.train_x[index*self.batch_size:(index+1)*self.batch_size]
            y = self.train_y[index*self.batch_size:(index+1)*self.batch_size]
            x, x_lengths = self.model.padMatrixWithoutTime(seqs=x)
            x_tensor = torch.from_numpy(x).float().to(self.device)
            if self.n_class == 2:
                y_tensor = torch.from_numpy(np.array(y)).float().to(self.device)
            else:
                y_tensor = torch.from_numpy(np.array(y)).long().to(self.device)
            pred = self.model(x_tensor)
            pred = pred.squeeze(1)
            loss = self.criterion(pred, y_tensor)
            loss.backward()
            self.optimizer.step()


            self.train_metrics.update('loss', loss.item())

            if index % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(index),
                    loss.item(),
                ))
                
            outs = torch.cat([outs, pred])
            trgs = torch.cat([trgs, y_tensor])

            
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(trgs, outs, self.n_class))
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch()
            log.update(**{'val_' + k: v for k, v in val_log.items()})


        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = torch.tensor([]).to(self.device)
            trgs = torch.tensor([]).to(dtype=torch.int64).to(self.device)
            for index in range(self.valid_n_batches):
                x = self.valid_x[index*self.batch_size:(index+1)*self.batch_size]
                y = self.valid_y[index*self.batch_size:(index+1)*self.batch_size]
                x, x_lengths = self.model.padMatrixWithoutTime(seqs=x)
                x_tensor = torch.from_numpy(x).float().to(self.device)
                if self.n_class == 2:
                    y_tensor = torch.from_numpy(np.array(y)).float().to(self.device)
                else:
                    y_tensor = torch.from_numpy(np.array(y)).long().to(self.device)
                pred = self.model(x_tensor)
                pred = pred.squeeze(1)
                loss = self.criterion(pred, y_tensor)
                self.valid_metrics.update('loss', loss.item())
                    
                outs = torch.cat([outs, pred])
                trgs = torch.cat([trgs, y_tensor])
                
        for met in self.metric_ftns:
            self.valid_metrics.update(met.__name__, met(trgs, outs, self.n_class))

        return self.valid_metrics.result()
        
        
    def _test_epoch(self):
        PATH = str(self.checkpoint_dir / 'model_best.pth')
        self.model.load_state_dict(torch.load(PATH)['state_dict'])
        self.model.eval()
        
        val_log = self._valid_epoch()
        
        self.test_ICD9_metrics.reset()
        with torch.no_grad():
            outs = torch.tensor([]).to(self.device)
            trgs = torch.tensor([]).to(dtype=torch.int64).to(self.device)
            for index in range(self.test_ICD9_n_batches):
                x = self.test_ICD9_x[index*self.batch_size:(index+1)*self.batch_size]
                y = self.test_ICD9_y[index*self.batch_size:(index+1)*self.batch_size]
                x, x_lengths = self.model.padMatrixWithoutTime(seqs=x)
                x_tensor = torch.from_numpy(x).float().to(self.device)
                if self.n_class == 2:
                    y_tensor = torch.from_numpy(np.array(y)).float().to(self.device)
                else:
                    y_tensor = torch.from_numpy(np.array(y)).long().to(self.device)
                pred = self.model(x_tensor)
                pred = pred.squeeze(1)
                loss = self.criterion(pred, y_tensor)
                self.test_ICD9_metrics.update('loss', loss.item())
                    
                outs = torch.cat([outs, pred])
                trgs = torch.cat([trgs, y_tensor])
                
        for met in self.metric_ftns:
            self.test_ICD9_metrics.update(met.__name__, met(trgs, outs, self.n_class))
        test_ICD9_log = self.test_ICD9_metrics.result()
        
        ###############################################################################
        
        self.test_ICD10_metrics.reset()
        with torch.no_grad():
            outs = torch.tensor([]).to(self.device)
            trgs = torch.tensor([]).to(dtype=torch.int64).to(self.device)
            for index in range(self.test_ICD10_n_batches):
                x = self.test_ICD10_x[index*self.batch_size:(index+1)*self.batch_size]
                y = self.test_ICD10_y[index*self.batch_size:(index+1)*self.batch_size]
                x, x_lengths = self.model.padMatrixWithoutTime(seqs=x)
                x_tensor = torch.from_numpy(x).float().to(self.device)
                if self.n_class == 2:
                    y_tensor = torch.from_numpy(np.array(y)).float().to(self.device)
                else:
                    y_tensor = torch.from_numpy(np.array(y)).long().to(self.device)
                pred = self.model(x_tensor)
                pred = pred.squeeze(1)
                loss = self.criterion(pred, y_tensor)
                self.test_ICD10_metrics.update('loss', loss.item())
                    
                outs = torch.cat([outs, pred])
                trgs = torch.cat([trgs, y_tensor])
            
        for met in self.metric_ftns:
            self.test_ICD10_metrics.update(met.__name__, met(trgs, outs, self.n_class))
        test_ICD10_log = self.test_ICD10_metrics.result()

        
        log = {}
        log.update(**{'val_' + k: v for k, v in val_log.items()})
        log.update(**{'test_ICD9_' + k: v for k, v in test_ICD9_log.items()})
        log.update(**{'test_ICD10_' + k: v for k, v in test_ICD10_log.items()})
        
        self.logger.info('='*100)
        self.logger.info('Test is completed')
        self.logger.info('-'*100)
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))            
            
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx * self.batch_size
        total = self.n_samples 

        return base.format(current, total, 100.0 * current / total)
