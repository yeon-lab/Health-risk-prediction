import pickle
import pandas as pd
import random
import numpy as np
from utils.util import split_data, split_data_DA, split_data_DG, Convert_to_numeric
SEED = 1111
random.seed(SEED)
np.random.seed(SEED)

def init_data(data_file, config):
    cohort = pickle.load(open(data_file, 'rb'))
    cohort = pd.DataFrame(cohort)
    cohort = cohort[(cohort.n_visit>=config['hyper_params']["min_visit"]) & (cohort.n_visit<=config['hyper_params']["max_visit"])]
    
    if config['hyper_params']['version']  == 'DG':
        if config['hyper_params']['model'] == 'AdaDiag':
            trainset, reweight, test = split_dataset_DA(all_data)
        else:
            trainset, reweight, test = split_dataset_DG(all_data)
            config["hyper_params"]["n_domains"] = 5  
    else:
        trainset, reweight, test = split_data(cohort)
        
    trainset = trainset.sample(frac=1, random_state=SEED)
    reweight = reweight.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test = test.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n_valid = int(len(trainset)*config['hyper_params']['valid_ratio'])
    train = trainset.iloc[n_valid:,:].reset_index(drop=True)
    valid = trainset.iloc[:n_valid,:].reset_index(drop=True)
    
    train_dates = train.last_observed_date.unique()
    valid_dates = valid.last_observed_date.unique()
    reweight_dates = reweight.last_observed_date.unique()
    test_dates = test.last_observed_date.unique()
    
    print('[train] min date: {}, max date: {}'.format(min(train_dates), max(train_dates)))
    print('[valid] min date: {}, max date: {}'.format(min(valid_dates), max(valid_dates)))
    print('[reweight] min date: {}, max date: {}'.format(min(reweight_dates), max(reweight_dates)))
    print('[test] min date: {}, max date: {}'.format(min(test_dates), max(test_dates)))
    
    print('\n# train: {}, # valid: {}, # test: {}, # reweight_test: {}'.format(
        len(train),
        len(valid),
        len(test),
        len(reweight)))
    
    print('\nAvg. # of visits for train: {}, for valid: {}, for test: {}, for reweight: {}\n'.format(
        np.mean(train.n_visit),
        np.mean(valid.n_visit),
        np.mean(test.n_visit),
        np.mean(reweight.n_visit)))
    
    
    convert_numeric = Convert_to_numeric()
    train = convert_numeric.forward(train.to_dict('list'), is_train=True)
    valid = convert_numeric.forward(valid.to_dict('list'))
    test = convert_numeric.forward(test.to_dict('list'))
    reweight = convert_numeric.forward(reweight.to_dict('list'))
    
    dataset ={
        'train': train,
        'valid': valid,
        'test': test,
        'reweight': reweight
    }

    return dataset, convert_numeric.feat_dict, convert_numeric.y_dict
