import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import os.path
import random

import torch
import torch.nn as nn

SEED = 1111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_npy(path):
    case_list = list()
    control_list = list()
    for root, dirs, files in os.walk(path):
        for file in files:
            if "case" in file:
                case_list.append(os.path.join(root, file))
            if "control" in file:
                control_list.append(os.path.join(root, file))
    print('case_list len:', len(case_list))   
    print('control_list len:', len(control_list))   
    for i, (f_case, f_control) in enumerate(zip(case_list, control_list)):
        if i == 0:
            all_cases = np.load(f_case, allow_pickle=True).item()
            all_controls = np.load(f_control, allow_pickle=True).item()
        else:
            case = np.load(f_case, allow_pickle=True).item()
            control = np.load(f_control, allow_pickle=True).item()
            for key in all_cases.keys():
                all_cases[key].extend(case[key])
            for key in all_controls.keys():
                all_controls[key].extend(control[key])

    print('# cases:', len(all_cases['DX']))
    print('# controls:', len(all_controls['DX']))
    
    return all_cases, all_controls
    

    
class Convert_to_numeric:
    def __init__(self, config):
        super(Convert_to_numeric, self).__init__()
        self.feat_idx = 0
        self.feat_dict = {}
        self.y_dict = None
        
    def transform(self, seqs, is_train):
        code_list = list()
        for sub_seq in seqs:
            sub_list=list()
            for code in sub_seq:
                if is_train and code not in self.feat_dict.keys():
                    self.feat_dict[code] = self.feat_idx
                    self.feat_idx += 1
                if code in self.feat_dict.keys():
                    sub_list.append(self.feat_dict[code])
            code_list.append(sub_list)
        return code_list
        

    def forward(self, data, is_train=False):
        control_y = 'Normal'
        if is_train:
            y_dict_lst = [control_y]+list(set(data['Y']))
            self.y_dict = {y_dict_lst[i]: i for i in range(len(y_dict_lst))}

        data['control_Y'] = [pd.NA for _ in range(len(data['DX']))]
        for idx in range(len(data['DX'])):
            case_dxs = data['DX'][idx]
            control_dxs = data['control_DX'][idx]
    
            data['DX'][idx] = self.transform(case_dxs, is_train)
            data['n_visit'][idx] = len(data['DX'][idx])
            data['Y'][idx] = self.y_dict[data['Y'][idx]]
            
            if control_dxs is not pd.NA:
                data['control_DX'][idx] = self.transform(control_dxs, is_train)
                data['control_Y'][idx] = self.y_dict[control_y]
            
        return pd.DataFrame(data)
    

def set_input_data(df):
    df = df.dropna(subset=['control_DX'], axis=0)
    case_df = df[['DX', 'Y', 'last_observed_date']]
    control_df = df[['control_DX', 'control_Y', 'control_last_observed_date']]
    control_df = control_df.rename(columns={'control_DX':'DX', 'control_Y':'Y', 'control_last_observed_date':'last_observed_date'})

    all_df = pd.concat([case_df, control_df], axis=0).sample(frac=1, random_state=SEED)
    print(f'# df: {len(df)}, # case: {len(case_df)}, # control: {len(control_df)}, # all: {len(all_df)}')
    print('all_df value_counts:\n', all_df.Y.value_counts())
    return all_df
    
    
def get_controls(all_cases, all_controls):
    n_samples = len(all_cases['DX'])
    all_cases['control_DX'] = [pd.NA for _ in range(n_samples)]
    all_cases['control_DX_f'] = [pd.NA for _ in range(n_samples)]
    all_cases['control_last_observed_date'] = [pd.NA for _ in range(n_samples)]
    print(pd.DataFrame(all_cases))
    
    all_controls = pd.DataFrame(all_controls)
    print(all_controls)
    
    for case_idx in tqdm(range(n_samples), total=n_samples):
        n_visit = all_cases['n_visit'][case_idx]
        year = all_cases['year'][case_idx]
        month = all_cases['month'][case_idx]
        age = all_cases['age'][case_idx]
        gender = all_cases['gender'][case_idx]
        indices = all_controls[(all_controls.n_visit >= n_visit) & (all_controls.year == year) & (all_controls.month == month) &\
                                    (all_controls.age == age) & (all_controls.gender == gender)].index
                                    
        if len(indices) > 0:
            control_idx = indices[0]
            control_DX = all_controls.loc[control_idx,'DX']
            control_DX_f = all_controls.loc[control_idx,'DX_f']
            control_last_observed_date = all_controls.loc[control_idx,'last_observed_date']
            
            all_cases['control_DX'][case_idx] = control_DX[len(control_DX)-n_visit:]
            all_cases['control_DX_f'][case_idx] = control_DX_f[len(control_DX_f)-n_visit:]
            all_cases['control_last_observed_date'][case_idx] = control_last_observed_date
            all_controls = all_controls.drop(control_idx, axis = 0)
            
    return all_cases

    
def init_data(data_file, npy_path, config):
    if os.path.exists(data_file):
        print('load existing data file')
        all_data = np.load(data_file, allow_pickle=True).item()
    else:
        print('No existing file. Save data file')
        all_cases, all_controls = load_npy(npy_path)
        all_data = get_controls(all_cases, all_controls)
        np.save(data_file, all_data) 
        print('saved data file')

    max_visit = config["hyper_params"]["max_visit"]
    min_visit = config["hyper_params"]["min_visit"]
    
    all_data = pd.DataFrame(all_data)
    all_data = all_data[(all_data.n_visit>=min_visit) & (all_data.n_visit<=max_visit)]
    
    trainset = pd.DataFrame()
    reweight_test = pd.DataFrame()
    test = pd.DataFrame()
    for date, subset in all_data.set_index('last_observed_date').groupby(pd.Grouper(freq='M')):
        year = date.strftime("%Y")
        month = date.strftime("%m")

        subset['last_observed_date'] = subset.index
        if year in ['2012','2013', '2014']:
            trainset = pd.concat([trainset, subset], axis=0)
        elif year == '2015' and month not in ['10', '11', '12']:
            trainset = pd.concat([trainset, subset], axis=0)
        elif year == '2015' and month in ['10', '11', '12']:
            reweight_test = pd.concat([reweight_test, subset], axis=0)
        elif year in ['2016','2017']:
            test = pd.concat([test, subset], axis=0)
    
        #print('# of {}: {}'.format(year, len()))
        
    trainset = trainset.sample(frac=1, random_state=SEED)
    n_train = int(len(trainset)*0.6)
    train = trainset.iloc[:n_train,:]
    valid = trainset.iloc[n_train:,:]

    train_dates = train.last_observed_date.unique()
    valid_dates = valid.last_observed_date.unique()
    test_dates = test.last_observed_date.unique()
    reweight_test_dates = reweight_test.last_observed_date.unique()
    
    print('[train] min date: {}, max date: {}'.format(min(train_dates), max(train_dates)))
    print('[valid] min date: {}, max date: {}'.format(min(valid_dates), max(valid_dates)))
    print('[test] min date: {}, max date: {}'.format(min(test_dates), max(test_dates)))
    print('[reweight_test] min date: {}, max date: {}'.format(min(reweight_test_dates), max(reweight_test_dates)))

    print('\n# train: {}, # valid: {}, # test: {}, # reweight_test: {}'.format(
        len(train),
        len(valid),
        len(test),
        len(reweight_test)))
        
    print('\nAvg. # of visits for train case: {}, for valid case: {}, for test case: {}, for reweight_test case: {}\n'.format(
        np.mean(train.n_visit),
        np.mean(valid.n_visit),
        np.mean(test.n_visit),
        np.mean(reweight_test.n_visit)))
        
    convert_numeric = Convert_to_numeric(config)
    train = convert_numeric.forward(train.reset_index(drop=True).to_dict('series'), is_train=True)
    valid = convert_numeric.forward(valid.reset_index(drop=True).to_dict('series'))
    test = convert_numeric.forward(test.reset_index(drop=True).to_dict('series'))
    reweight_test = convert_numeric.forward(reweight_test.reset_index(drop=True).to_dict('series'))
    print('train feat_dict len:', len(convert_numeric.feat_dict.keys()))
    print('\ny_dict:', convert_numeric.y_dict)

    
    print('\ntrain','-'*100)
    train = set_input_data(train)
    print('\nvalid','-'*100)
    valid = set_input_data(valid)
    print('\ntest','-'*100)
    test = set_input_data(test)
    print('\nreweight_test','-'*100)
    reweight_test = set_input_data(reweight_test)

    train_dates = train.last_observed_date.unique()
    valid_dates = valid.last_observed_date.unique()
    test_dates = test.last_observed_date.unique()
    reweight_test_dates = reweight_test.last_observed_date.unique()

    dataset ={
        'train': train.to_dict('series'),
        'valid': valid.to_dict('series'),
        'test': test.to_dict('series'),
        'reweight_test': reweight_test.to_dict('series')
    }
    
    return dataset, convert_numeric.feat_dict, convert_numeric.y_dict

 
def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight.data, -0.1, 0.1)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif type(m) == nn.GRU or type(m) == nn.LSTM:
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.uniform_(m.__getattr__(p), -0.1, 0.1)
                if 'bias' in p:
                    torch.nn.init.constant_(m.__getattr__(p), 0.0)



def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader
        
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
        
