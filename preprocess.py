import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os 
from utils.util import *

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
                                    X_idx = j
                                    Y_idx = i
                                    case_group['ID'].append(id_)
                                    case_group['DXVER'].append(DXVER_)
                                    case_group['X'].append(X)
                                    case_group['Y'].append(Y)
                                    case_group['X_idx'].append(X_idx)
                                    case_group['Y_idx'].append(Y_idx)
                                    case_group['n_visit'].append(len(X))
                                    
                                    patient_demo = demo[demo.ENROLID == int(id_)]
                                    case_group['age'].append(patient_demo['DOBYR'].item())
                                    case_group['gender'].append(patient_demo['SEX'].item())

                    if is_saved:
                        break
                break
    

        if not is_case:
            for DXVER_ in ['9','0']:
                control_patient = patient[patient.DXVER==DXVER_]
                if len(control_patient.SVCDATE.unique()) >= n_visits:
                    X = get_dx_list(control_patient, DXVER_, mapping, DX_col)
                    Y = 'Normal'
                    control_group['ID'].append(id_)
                    control_group['DXVER'].append(DXVER_)
                    control_group['X'].append(X)
                    control_group['Y'].append(Y)
                    control_group['n_visit'].append(len(X))
                    
                    patient_demo = demo[demo.ENROLID == int(id_)]
                    case_group['age'].append(patient_demo['DOBYR'].item())
                    case_group['gender'].append(patient_demo['SEX'].item())

            
        
    np.save(os.path.join(config.save_dir, 'control_group'), control_group)  
    np.save(os.path.join(config.save_dir, 'case_group'), case_group) 
    
    


        
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--path', default = './data/',type=str)
    args.add_argument('--save_dir', default = './data_npy/',type=str)
    args.add_argument('--n_visits', default = 10,type=int)
    args.add_argument('--fold_id', type=int)
    
    config = args.parse_args()
    
    target_dict = {
        'HF': ['I11', 'I13', 'I50', 'I42', 'K77'],
        'ST': ['Z8673', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'G458', 'G459']

    }
    
    mapping = pd.read_csv('icd9to10.csv')[['icd9cm', 'icd10cm']]
    mapping = dict(mapping.values.tolist())
        
    df = load_raw(config.path)

    all_ids = df.ENROLID.unique()
    batch_size = len(all_ids)//config.n_split
    
    demo = pd.read_csv('demo.csv')
    
    get_cases(df, all_ids, target_dict, mapping, config, demo)
 
