import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os 
from utils.util import *


def get_controls(all_cases, all_controls):
    all_cases['control_DX'] = [pd.NA for _ in range(len(all_cases['DX']))]
    all_cases['control_last_observed_date'] = [pd.NA for _ in range(len(all_cases['DX']))]    
    all_controls = pd.DataFrame(all_controls)
    
    for case_idx in tqdm(range(len(all_cases['DX'])), total=len(all_cases['DX'])):
        n_visit = all_cases['n_visit'][case_idx]
        year = all_cases['year'][case_idx]
        month = all_cases['month'][case_idx]
        age = all_cases['age'][case_idx]
        gender = all_cases['gender'][case_idx]
        indices = all_controls[(all_controls.n_visit == n_visit) & (all_controls.year == year) & (all_controls.month == month) &\
                                    (all_controls.age == age) & (all_controls.gender == gender)].index      
                                        
        if len(indices) > 0:
            control_idx = indices[0]
            control_DX = all_controls.loc[control_idx,'DX']
            control_last_observed_date = all_controls.loc[control_idx,'last_observed_date']
            all_cases['control_DX'][case_idx] = control_DX[len(control_DX)-n_visit:]
            all_cases['control_last_observed_date'][case_idx] = control_last_observed_date
            all_controls = all_controls.drop(control_idx, axis = 0)
    return all_cases

def check_target(dx, target_dict, DXVER, mapping):
    if DXVER == '9':
        dx = icd9_to_icd10(dx, mapping)
        if dx is None:
            return False, None
            
    for key in target_dict.keys():
        for code in target_dict[key]:
            if dx.startswith(code):
                return True, key
    return False, None
    

def get_cases(df, all_ids, target_dict, mapping, config, demo):
    n_visits = config.n_visits

    DX_col = [col for col in df.columns if 'DX' in col and col != 'DXVER']
    
    case_group = {
        'ID':[],
        'DXVER': [],
        'X': [],
        'Y': [],
        'n_visit': [],
        'X_idx': [],
        'Y_idx': [],
        'age': [],
        'gender':[]
    }

    control_group = {
        'ID':[],
        'DXVER': [],
        'X': [],
        'Y': [],
        'n_visit': [],
        'age': [],
        'gender':[]
    }
    
    control_group_ids = []        
    for id_ in tqdm(all_ids, total=len(all_ids)):
        is_saved = False
        patient = df[df.ENROLID == id_].reset_index(drop=True)
        
        if len(patient[patient.DXVER == '9'].SVCDATE.unique()) < n_visits:
            continue
        elif len(patient[patient.DXVER == '0'].SVCDATE.unique()) < n_visits:
            continue
            
        for i, (_, row) in enumerate(patient.iterrows()):
            DXVER = row['DXVER']
    
            dxs = list(row[DX_col])
            for dx in dxs:
                if not pd.isnull(dx):
                    is_case, target = check_target(dx, target_dict, DXVER, mapping)
                    if is_case:
                        break
    
            if is_case and i < n_visits:
                break
            if is_case and i >= n_visits:
                if (row['SVCDATE']-patient.SVCDATE[n_visits-1]).days < 365:
                    break
                for j in range(i-1, n_visits-2, -1):
                    if (row['SVCDATE']-patient.SVCDATE[j]).days > 365:
                        patient_ = patient.iloc[:j+1,:]
                        for DXVER_ in ['9','0']:
                            patient_x = patient_[patient_.DXVER==DXVER_]
                            if len(patient_x.SVCDATE.unique()) >= n_visits:
                                X = get_dx_list(patient_x, DXVER_, mapping, DX_col)
                                if len(X) >= n_visits:
                                    is_saved = True
                                    Y = target
class Search_cases:
    def __init__(self, target_dict, mapping, config, demo, df, windows=[90, 180, 360], DXVER_list=['0','9']):
        super(Search_cases, self).__init__()
        self.target_dict = target_dict
        self.mapping = mapping
        self.demo = demo
        self.config = config
        self.n_visits = config.n_visits
        self.DX_col = [col for col in df.columns if 'DX' in col and col != 'DXVER']
        self.target = config.target
   
        self.windows = windows
        self.DXVER_list = DXVER_list
        self.case_group = dict()
        self.control_group = dict()
        for window in self.windows:
            self.case_group[window] = {
                'ID':[],
                'DX': [],
                'DATE': [],
                'Y': [],
                'n_visit': [],
                'last_observed_date': [],
                'year': [],
                'month': [],
                'age': [],
                'gender':[]
            }
            self.control_group[window] = {
                'ID':[],
                'DX': [],
                'DATE': [],
                'last_observed_date': [],
                'year': [],
                'month': [],
                'n_visit': [],
                'age': [],
                'gender':[]
            }  

            

    def forward(self, all_ids):
        for id_ in tqdm(all_ids, total=len(all_ids)):
            patient = df[df.ENROLID == id_].reset_index(drop=True)
            if len(patient.SVCDATE.unique()) < self.n_visits+1:
                continue
            
            patient_demo = self.demo[self.demo.ENROLID == id_]
            for i, (_, row) in enumerate(patient.iterrows()):
                DXVER = row['DXVER']
                SVCDATE = row['SVCDATE']
                dxs = list(row[self.DX_col])
                for dx in dxs:
                    if not pd.isnull(dx):
                        is_case, target = check_target(dx, self.target_dict, DXVER, self.mapping)
                        if is_case:
                            break
                if is_case and i < self.n_visits:
                    break
                if is_case and i >= self.n_visits:
                    patient_x = patient.iloc[:i,:]
                    observed_dates = patient_x.SVCDATE.unique()[self.n_visits-1:]
                    window_list = self.windows.copy()
                    unsaved_window_list = self.windows.copy()
                    save_date = dict()
                    for date_idx, last_observed_date in enumerate(reversed(observed_dates)):
                        days = (SVCDATE-last_observed_date).days
                        for window in window_list:
                            if days >= window:
                                save_date[window] = {'date_idx': date_idx,
                                                     'last_observed_date': pd.to_datetime(last_observed_date)}
                                unsaved_window_list.remove(window)
                        
                        if len(save_date) == len(self.windows):
                            break
                        else:
                            window_list = unsaved_window_list.copy()
                            
                    if len(save_date) > 0: 
                        DX_list, DATE_list = get_code_list(patient_x, self.mapping, self.DX_col)
                        for window, date in save_date.items():
                            if date['date_idx'] == 0:
                                DX = DX_list
                                DATE = DATE_list
                            else:
                                DX = DX_list[:-date['date_idx']]
                                DATE = DATE_list[:-date['date_idx']]
                            
                            self.case_group[window]['ID'].append(id_)
                            self.case_group[window]['DX'].append(DX)
                            self.case_group[window]['DATE'].append(DATE)
                            self.case_group[window]['Y'].append(target)
                            self.case_group[window]['n_visit'].append(len(DX))
                            self.case_group[window]['last_observed_date'].append(date['last_observed_date'])
                            self.case_group[window]['year'].append(date['last_observed_date'].strftime('%Y'))
                            self.case_group[window]['month'].append(date['last_observed_date'].strftime('%m'))
                            self.case_group[window]['age'].append(patient_demo['DOBYR'].item())
                            self.case_group[window]['gender'].append(patient_demo['SEX'].item())
                    break
                
            if not is_case:
                SVCDATE= patient.SVCDATE.tolist()[-1]
                patient_x = patient.iloc[:-1,:]
                observed_dates = patient_x.SVCDATE.unique()[self.n_visits-1:]
                window_list = self.windows.copy()
                unsaved_window_list = self.windows.copy()
                save_date = dict()
                for date_idx, last_observed_date in enumerate(reversed(observed_dates)):
                    days = (SVCDATE-last_observed_date).days
                    for window in window_list:
                        if days > window:
                            save_date[window] = {'date_idx': date_idx,
                                                 'last_observed_date': pd.to_datetime(last_observed_date)}
                            unsaved_window_list.remove(window)
                    
                    if len(save_date) == len(self.windows):
                        break
                    else:
                        window_list = unsaved_window_list.copy()
                        
                if len(save_date) > 0: 
                    DX_list, DATE_list = get_code_list(patient_x, self.mapping, self.DX_col)
                    for window, date in save_date.items():
                        if date['date_idx'] == 0:
                            DX = DX_list
                            DATE = DATE_list
                        else:
                            DX = DX_list[:-date['date_idx']]
                            DATE = DATE_list[:-date['date_idx']]

                        self.control_group[window]['ID'].append(id_)
                        self.control_group[window]['DX'].append(DX)
                        self.control_group[window]['DATE'].append(DATE)
                        self.control_group[window]['n_visit'].append(len(DX))
                        self.control_group[window]['last_observed_date'].append(date['last_observed_date'])
                        self.control_group[window]['year'].append(date['last_observed_date'].strftime('%Y'))
                        self.control_group[window]['month'].append(date['last_observed_date'].strftime('%m'))
                        self.control_group[window]['age'].append(patient_demo['DOBYR'].item())
                        self.control_group[window]['gender'].append(patient_demo['SEX'].item())


            
       
        for window in self.windows:
            all_data = get_controls(self.case_group[window], self.control_group[window])
            np.save('input_{}_{}.npy'.format(self.target, window), all_data) 


        
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--path', default = './data/',type=str)
    args.add_argument('--n_visits', default = 10,type=int)
    args.add_argument('--target', default = 'hf',type=str)
    
    config = args.parse_args()
    if config.target == 'hf':
        target_dict = {
            'HF': ['I11', 'I13', 'I50', 'I42', 'K77']
        }
    else:
        target_dict = {
            'ST': ['Z8673', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'G458', 'G459']
        }    
        
    mapping = pd.read_csv('icd9to10.csv')[['icd9cm', 'icd10cm']]
    mapping = dict(mapping.values.tolist())
        
    df = load_raw(config.path)

    all_ids = df.ENROLID.unique()
    
    demo = pd.read_csv('demo.csv')
    
    Load_case = Search_cases(target_dict, mapping, config, demo, df)
    Load_case.forward(all_ids)
    
 
