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

SEED = 1111



    
class Convert_to_numeric:
    def __init__(self):
        super(Convert_to_numeric, self).__init__()
        self.feat_dict_idx = 0
        self.feat_dict = {}
    
    def transform_dx(self, dx_list):
        code_list = list()
        for sub_seq in dx_list:
            sub_list=list()
            for code in sub_seq:
                if code not in self.feat_dict.keys():
                    self.feat_dict[code] = self.feat_dict_idx
                    self.feat_dict_idx += 1
                sub_list.append(self.feat_dict[code])
            code_list.append(sub_list)
        return code_list
    
    def forward(self, all_data):
        control_y = 'Normal'
        y_dict_lst = [control_y]+list(set(all_data['Y']))
        y_dict = {y_dict_lst[i]: i for i in range(len(y_dict_lst))}

        all_data['control_Y'] = [pd.NA for _ in range(len(all_data['X']))]
        for idx in all_data['X'].index:
            case_dxs = all_data['X'][idx]
            control_dxs = all_data['control_X'][idx]
            
            all_data['X'][idx] = self.transform_dx(case_dxs)
            all_data['Y'][idx] = y_dict[all_data['Y'][idx]]
            
            if control_dxs != pd.NA:
                all_data['control_X'][idx] = self.transform_dx(control_dxs)
                all_data['control_Y'][idx] = y_dict[control_y]
            
        return all_data, self.feat_dict, y_dict

def set_input_data(df):
    case_df = df[['X','Y']]
    control_df = df[['control_X','control_Y']].dropna(axis=0)
    control_df.rename(columns={'control_X':'X', 'control_Y':'Y'}, inplace=True)
    all_df = pd.concat([case_df, control_df], axis=0).sample(frac=1, random_state=SEED)
    print(all_df.Y.value_counts())
    print(f'# df: {len(df)}, # case: {len(case_df)}, # control: {len(control_df)}, # all: {len(all_df)}, DXVER: {df.DXVER.unique()}')

    return all_df
    

    
def init_data(data_file, config):
    
    all_data_ = np.load(data_file, allow_pickle=True).item()
        
    all_data_ = pd.DataFrame(all_data_)
    all_data_9 = all_data_[(all_data_.DXVER == '9') & (all_data_.n_visit <= config["hyper_params"]["max_visit"])]
    all_data_10 = all_data_[all_data_.DXVER == '0']
    all_data = pd.concat([all_data_9, all_data_10], axis=0).reset_index(drop=True).to_dict('series')  
    
    convert_numeric = Convert_to_numeric()
    all_data, feat_dict, y_dict = convert_numeric.forward(all_data)  
    print('y_dict:\n', y_dict)
    df_data = pd.DataFrame(all_data).sample(frac=1, random_state=SEED)
    test_ICD10 = df_data[df_data.DXVER == '0']
    ICD9 = df_data[df_data.DXVER == '9']
    test_ICD9 = ICD9.iloc[:len(test_ICD10),:]
    trainset = ICD9.iloc[len(test_ICD10):,:]
    train = trainset.iloc[len(test_ICD10):,:]
    vaild = trainset.iloc[:len(test_ICD10),:]
    print('# ICD-9: {}, # ICD-9 train: {}, # ICD-9 valid: {}, # ICD-9 test: {}, # ICD-10 test: {}'.format(
        len(ICD9),
        len(train),
        len(vaild),
        len(test_ICD9),
        len(test_ICD10)))

    print('Avg. # of visits for ICD-9 train case: {}, for ICD-9 valid case: {}, for ICD-9 test case: {}, for ICD-10 case: {}'.format(
        np.mean(train.n_visit),
        np.mean(vaild.n_visit),
        np.mean(test_ICD9.n_visit),
        np.mean(test_ICD10.n_visit)))
        
    train = set_input_data(train)
    vaild = set_input_data(vaild)
    test_ICD9 = set_input_data(test_ICD9)
    test_ICD10 = set_input_data(test_ICD10)        
    dataset ={
        'train': train.to_dict('series'),
        'valid': vaild.to_dict('series'),
        'test_ICD9': test_ICD9.to_dict('series'),
        'test_ICD10': test_ICD10.to_dict('series')
    }
    return dataset, feat_dict, y_dict



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
        
        
