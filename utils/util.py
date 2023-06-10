import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import os.path
import random

class Convert_to_numeric:
    def __init__(self, ):
        super(Convert_to_numeric, self).__init__()
        self.feat_idx = 0
        self.feat_dict = {}
        self.y_idx = 1
        self.y_dict = {'Normal': 0}
        
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
        for idx in range(len(data['DX'])):
            y = data['Y'][idx]
            if y not in self.y_dict.keys():
                self.y_dict[y] = self.y_idx
                self.y_idx += 1
            data['Y'][idx] = self.y_dict[y]
            data['DX'][idx] = self.transform(data['DX'][idx], is_train)
        return data
    
def split_data(all_data):
    trainset, reweight, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for date, subset in all_data.set_index('last_observed_date').groupby(pd.Grouper(freq='M')):
        year = date.strftime("%Y")
        month = date.strftime("%m")

        subset['last_observed_date'] = subset.index
        if year in ['2012', '2013', '2014']:
            trainset = pd.concat([trainset, subset], axis=0)
        elif year == '2015' and month not in ['10', '11', '12']:
            trainset = pd.concat([trainset, subset], axis=0)
        elif year == '2015' and month in ['10', '11', '12']:
            reweight = pd.concat([reweight, subset], axis=0)
        elif year in ['2016', '2017']:
            test = pd.concat([test, subset], axis=0)

    return trainset, reweight, test

def split_data_DA(all_data):
    trainset = pd.DataFrame()
    reweight = pd.DataFrame()
    test = pd.DataFrame()
    for date, subset in all_data.set_index('last_observed_date').groupby(pd.Grouper(freq='M')):
        year = date.strftime("%Y")
        month = date.strftime("%m")

        subset['last_observed_date'] = subset.index
        if year in ['2012','2013', '2014']:
            subset['Domain'] = 0
            trainset = pd.concat([trainset, subset], axis=0)
        elif year == '2015' and month not in ['10', '11', '12']:
            subset['Domain'] = 0
            trainset = pd.concat([trainset, subset], axis=0)
        elif year == '2015' and month in ['10', '11', '12']:
            subset['Domain'] = 1
            reweight = pd.concat([reweight, subset], axis=0)
        elif year in ['2016', '2017']:
            subset['Domain'] = 1
            test = pd.concat([test, subset], axis=0)
    trainset = pd.concat([trainset, reweight], axis=0)
    return trainset, reweight, test
    
def split_data_DG(all_data):
    trainset = pd.DataFrame()
    reweight = pd.DataFrame()
    test = pd.DataFrame()
    
    domain_dict = {
        '2012': 0,
        '2013': 1,
        '2014': 2,
        '2015': 3
    }
    for date, subset in all_data.set_index('last_observed_date').groupby(pd.Grouper(freq='M')):
        year = date.strftime("%Y")
        month = date.strftime("%m")

        subset['last_observed_date'] = subset.index
        if year in ['2012', '2013', '2014']:
            subset['Domain'] = domain_dict[year]
            trainset = pd.concat([trainset, subset], axis=0)
        elif year == '2015' and month not in ['10', '11', '12']:
            subset['Domain'] = domain_dict[year]
            trainset = pd.concat([trainset, subset], axis=0)
        elif year == '2015' and month in ['10', '11', '12']:
            subset['Domain'] = 4
            reweight = pd.concat([reweight, subset], axis=0)
        elif year in ['2016', '2017']:
            subset['Domain'] = 4
            test = pd.concat([test, subset], axis=0)

    trainset = pd.concat([trainset, reweight], axis=0)
    return trainset, reweight, test
    

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
        
