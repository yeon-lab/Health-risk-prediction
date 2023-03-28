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

pd.set_option('display.max_columns', None)
SEED = 1111

    

    
class Convert_to_numeric:
    def __init__(self):
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
    all_cases['control_DX'] = [pd.NA for _ in range(len(all_cases['DX']))]
    all_cases['control_last_observed_date'] = [pd.NA for _ in range(len(all_cases['DX']))]
    print(pd.DataFrame(all_cases))
    
    all_controls = pd.DataFrame(all_controls)
    print(all_controls)
    
    for case_idx in tqdm(range(len(all_cases['DX'])), total=len(all_cases['DX'])):
        n_visit = all_cases['n_visit'][case_idx]
        year = all_cases['year'][case_idx]
        month = all_cases['month'][case_idx]
        age = all_cases['age'][case_idx]
        gender = all_cases['gender'][case_idx]
        indices = all_controls[(all_controls.n_visit == n_visit) & (all_controls.year == year) & (all_controls.month == month) &\
                                    (all_controls.age == age) & (all_controls.gender == gender)].index
                                    
        if len(indices) == 0:
            indices = all_controls[(all_controls.n_visit >= n_visit) & (all_controls.year == year) & (all_controls.month == month) &\
                                        (all_controls.age == age) & (all_controls.gender == gender)].index  
                                        
        if len(indices) == 0 and year in ['2012', '2013', '2014', '2016']:
            indices = all_controls[(all_controls.n_visit >= n_visit) & (all_controls.year == year) &\
                                        (all_controls.age == age) & (all_controls.gender == gender)].index  
                                        
        if len(indices) == 0 and year == '2015' and month in ['10','11','12']:
            indices = all_controls[(all_controls.n_visit >= n_visit) & (all_controls.year == '2016') &\
                                        (all_controls.age == age) & (all_controls.gender == gender)].index  
                                        
        if len(indices) == 0 and year == '2015' and month not in ['10','11','12']:
            indices = all_controls[(all_controls.n_visit >= n_visit) & (all_controls.year == '2014') &\
                                        (all_controls.age == age) & (all_controls.gender == gender)].index  
                                        
        if len(indices) > 0:
            control_idx = indices[0]
            control_DX = all_controls.loc[control_idx,'DX']
            control_last_observed_date = all_controls.loc[control_idx,'last_observed_date']
            all_cases['control_DX'][case_idx] = control_DX[len(control_DX)-n_visit:]
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
    
    pre_train = pd.DataFrame()
    trainset = pd.DataFrame()
    for date, subset in all_data.set_index('last_observed_date').groupby(pd.Grouper(freq='M')):
        year = date.strftime("%Y")
        month = date.strftime("%m")
        subset['last_observed_date'] = subset.index
        if year in ['2012','2013', '2014']:
            pre_train = pd.concat([pre_train, subset], axis=0)
        elif year == '2015' and month not in ['12']:
            pre_train = pd.concat([pre_train, subset], axis=0)
        elif year == '2015' and month in ['12']:
            trainset = pd.concat([trainset, subset], axis=0)
        elif year in ['2016', '2017']:
            trainset = pd.concat([trainset, subset], axis=0)
    
        #print('# of {}: {}'.format(year, len()))
        
    trainset = trainset.sample(frac=1, random_state=SEED)
    n_train = int(len(trainset)*0.6)
    post_train = trainset.iloc[:n_train,:]
    validset = trainset.iloc[n_train:,:]
    n_valid = int(len(validset)/2)
    valid = validset.iloc[:n_valid,:]
    test = validset.iloc[n_valid:,:]
    
    post_train_dates = post_train.last_observed_date.unique()
    valid_dates = valid.last_observed_date.unique()
    test_dates = test.last_observed_date.unique()
    pre_train_dates = pre_train.last_observed_date.unique()

    post_train_control_dates = post_train.control_last_observed_date.dropna().unique()
    valid_control_dates = valid.control_last_observed_date.dropna().unique()
    test_control_dates = test.control_last_observed_date.dropna().unique()
    pre_control_train_dates = pre_train.control_last_observed_date.dropna().unique()

    
    print('[post_train] min date: {}, max date: {}'.format(min(post_train_dates), max(post_train_dates)))
    print('[valid] min date: {}, max date: {}'.format(min(valid_dates), max(valid_dates)))
    print('[test] min date: {}, max date: {}'.format(min(test_dates), max(test_dates)))
    print('[pre_train] min date: {}, max date: {}'.format(min(pre_train_dates), max(pre_train_dates)))

    print('[post_train control] min date: {}, max date: {}'.format(min(post_train_control_dates), max(post_train_control_dates)))
    print('[valid control] min date: {}, max date: {}'.format(min(valid_control_dates), max(valid_control_dates)))
    print('[test control] min date: {}, max date: {}'.format(min(test_control_dates), max(test_control_dates)))
    print('[pre_train control] min date: {}, max date: {}'.format(min(pre_control_train_dates), max(pre_control_train_dates)))

    
    print('\n# post_train: {}, # valid: {}, # test: {}, # pre_train: {}'.format(
        len(post_train),
        len(valid),
        len(test),
        len(pre_train)))
        
    print('\nAvg. # of visits for post_train case: {}, for valid case: {}, for test case: {}, for pre_train case: {}\n'.format(
        np.mean(post_train.n_visit),
        np.mean(valid.n_visit),
        np.mean(test.n_visit),
        np.mean(pre_train.n_visit)))
        
    convert_numeric = Convert_to_numeric()
    if config['hyper_params']['version'] != 'raw':
        pre_train = convert_numeric.forward(pre_train.reset_index(drop=True).to_dict('series'), is_train=True)
        post_train = convert_numeric.forward(post_train.reset_index(drop=True).to_dict('series'))
    else:
        post_train = convert_numeric.forward(post_train.reset_index(drop=True).to_dict('series'), is_train=True)
    valid = convert_numeric.forward(valid.reset_index(drop=True).to_dict('series'))
    test = convert_numeric.forward(test.reset_index(drop=True).to_dict('series'))
    print('train feat_dict len:', len(convert_numeric.feat_dict.keys()))
    print('\ny_dict:', convert_numeric.y_dict)

    print('\nAvg. # of visits for post_train case: {}, for valid case: {}, for test case: {}, for pre_train case: {}'.format(
        np.mean(post_train.n_visit),
        np.mean(valid.n_visit),
        np.mean(test.n_visit),
        np.mean(pre_train.n_visit)))

    print('Min. # of visits for post_train case: {}, for valid case: {}, for test case: {}, for pre_train case: {}'.format(
        np.min(post_train.n_visit),
        np.min(valid.n_visit),
        np.min(test.n_visit),
        np.min(pre_train.n_visit)))

    print('Max. # of visits for post_train case: {}, for valid case: {}, for test case: {}, for pre_train case: {}\n'.format(
        np.max(post_train.n_visit),
        np.max(valid.n_visit),
        np.max(test.n_visit),
        np.min(pre_train.n_visit)))
    
    print('\ntrain','-'*100)
    post_train = set_input_data(post_train)
    print('\nvalid','-'*100)
    valid = set_input_data(valid)
    print('\ntest','-'*100)
    test = set_input_data(test)
    if config['hyper_params']['version'] != 'raw':
        print('\npost_train','-'*100)
        pre_train = set_input_data(pre_train)

    post_train_dates = post_train.last_observed_date.unique()
    valid_dates = valid.last_observed_date.unique()
    test_dates = test.last_observed_date.unique()
    print('[post_train] min date: {}, max date: {}'.format(min(post_train_dates), max(post_train_dates)))
    print('[valid] min date: {}, max date: {}'.format(min(valid_dates), max(valid_dates)))
    print('[test] min date: {}, max date: {}'.format(min(test_dates), max(test_dates)))
    if config['hyper_params']['version'] == 'weight':
        pre_train_dates = pre_train.last_observed_date.unique()
        print('[pre_train] min date: {}, max date: {}'.format(min(pre_train_dates), max(pre_train_dates)))
    
    if config['hyper_params']['version'] != 'raw':
        dataset ={
            'post_train': post_train.to_dict('series'),
            'valid': valid.to_dict('series'),
            'test': test.to_dict('series'),
            'pre_train': pre_train.to_dict('series')
        }
    else:
        dataset ={
            'post_train': post_train.to_dict('series'),
            'valid': valid.to_dict('series'),
            'test': test.to_dict('series')
        }    
    return dataset, convert_numeric.feat_dict, convert_numeric.y_dict



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
        
