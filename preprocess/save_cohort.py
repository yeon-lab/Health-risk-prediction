from collections import defaultdict
from datetime import datetime
import os
from tqdm import tqdm
import pandas as pd
import pickle

def save_input_cohort(args, user_dx, case_cohort, demo, target):
    pred_windows = args.pred_windows
    
    saved_cohort = dict()
    for window in pred_windows:
        saved_cohort[window] = defaultdict(list)
    
    for ID, date_dxs in tqdm(user_dx.items()):
        operation_date = case_cohort[target].get(ID)
        if operation_date is not None:
            operation_date = min(operation_date)
            Y = target
        else:
            operation_date = max(date_dxs.keys())
            Y = 'Normal'
    
        DX_list = defaultdict(list)
        DATE_list = defaultdict(list)
        for date, dx_code in sorted(date_dxs.items(), key= lambda x:x[0]):
            for window in pred_windows:
                if (operation_date - date).days >= window:
                    DX_list[window].append(list(set(dx_code)))
                    DATE_list[window].append(date)
                    
        for window, DX in DX_list.items():
            if len(DX) > args.min_visits:
                patient_demo = demo[demo.ENROLID == int(ID)]
                last_observed_date = DATE_list[window][-1]
                
                saved_cohort[window]['ID'].append(ID)
                saved_cohort[window]['DX'].append(DX)
                saved_cohort[window]['Y'].append(Y)
                saved_cohort[window]['DATE'].append(DATE_list[window])
                saved_cohort[window]['n_visit'].append(len(DX))
                saved_cohort[window]['last_observed_date'].append(last_observed_date)
                saved_cohort[window]['year'].append(last_observed_date.year)
                saved_cohort[window]['month'].append(last_observed_date.month)
                saved_cohort[window]['operation_date'].append(operation_date)
                saved_cohort[window]['age'].append(patient_demo.SEX.item())
                saved_cohort[window]['gender'].append(patient_demo.DOBYR.item())
                

    for window, saved_dict in saved_cohort.items():
        file_name = os.path.join(args.pkl_dir, '{}_saved_cohort_{}.pkl'.format(target, window))                
        pickle.dump(saved_dict, open(file_name, 'wb'))
        print(f'dumped {file_name}...', flush=True)  
                
