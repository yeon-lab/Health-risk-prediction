import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob


def load_data(path = './data_npy/'):
    case_list = list()
    control_list = list()
    for root, dirs, files in os.walk(path):
        for file in files:
            if "case" in file:
                case_list.append(os.path.join(root, file))
            if "control" in file:
                control_list.append(os.path.join(root, file))
                
    all_controls = list()
    for i, (f_case, f_control) in enumerate(zip(case_list, control_list)):
        if i == 0:
            all_cases = np.load(f_case, allow_pickle=True).item()
        else:
            case = np.load(f_case, allow_pickle=True).item()
            for key in all_cases.keys():
                all_cases[key].extend(case[key])
        all_controls.extend(np.load(f_control))

    print('case len:', len(all_cases['X']))
    print('control len:', len(all_controls))
    return all_cases, all_controls
    


def convert_to_numeric(all_cases):
    feat_dict = {}
    i = 0
    for sublist in all_cases['X']:
        codes = sum(sublist , [])
        for code in codes:
            if code not in feat_dict.keys():
                feat_dict[code] = i
                i += 1
    
    y_dict_lst = list(set(all_cases['Y']))
    y_dict = {y_dict_lst[i]: i for i in range(len(y_dict_lst))}

    print(all_cases['Y'].value_counts())
    print(y_dict)
    
    for idx in all_cases['X'].index:
        all_codes = all_cases['X'][idx]
        code_list = list()
        for sub_seq in all_codes:
            sub_list=list()
            for code in sub_seq:
                sub_list.append(feat_dict[code])
            code_list.append(sub_list)
        all_cases['X'][idx] = code_list
        all_cases['Y'][idx] = y_dict[all_cases['Y'][idx]]
        
    return all_cases, feat_dict, y_dict
    
    
def init_data(path='./data_npy/'):
    all_cases, all_controls = load_data(path)
    all_cases['n_visit'] = np.zeros(len(all_cases['X'])).astype(int)
    for i in range(len(all_cases['n_visit'])):
        all_cases['n_visit'][i] = len(all_cases['X'][i])
        
    all_cases = pd.DataFrame(all_cases)
    all_cases = all_cases[all_cases.n_visit < 100].to_dict('series')  

    all_cases, feat_dict, y_dict = convert_to_numeric(all_cases)  
    
    df_case = pd.DataFrame(all_cases).sample(frac=1)
    test_case_ICD10 = df_case[df_case.DXVER == '0']
    ICD9 = df_case[df_case.DXVER == '9'].reset_index(drop=True)
    test_case_ICD9 = ICD9.iloc[:len(test_case_ICD10),:]
    trainset = ICD9.iloc[len(test_case_ICD10):,:].reset_index(drop=True)
    train_ratio = round(len(trainset)*0.9)
    train_case = trainset.iloc[len(test_case_ICD10):,:]
    vaild_case = trainset.iloc[:len(test_case_ICD10),:]
    print('# ICD-9: {}, # ICD-9 train case: {}, # ICD-9 valid case: {}, # ICD-9 test case: {}, # ICD-10 case: {}'.format(
        len(ICD9),
        len(train_case),
        len(vaild_case),
        len(test_case_ICD9),
        len(test_case_ICD10)))

    print('Avg. # of visits for ICD-9 train case: {}, for ICD-9 valid case: {}, for ICD-9 test case: {}, for ICD-10 case: {}'.format(
        np.mean(train_case.n_visit),
        np.mean(vaild_case.n_visit),
        np.mean(test_case_ICD9.n_visit),
        np.mean(test_case_ICD10.n_visit)))
        

    dataset ={
        'train_case': train_case.to_dict('series'),
        'valid_case': vaild_case.to_dict('series'),
        'test_case_ICD9': test_case_ICD9.to_dict('series'),
        'test_case_ICD10': test_case_ICD10.to_dict('series')
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
        
        