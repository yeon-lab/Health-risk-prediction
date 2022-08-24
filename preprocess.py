import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os 

    
def load_dataset(path='./data/'):
    outpatient_files = os.listdir(path)

    files = [os.path.join(path, file) for file in outpatient_files]
    
    data = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, dtype=str)
        df = df[~df['DX1'].isnull()]
        df = df.drop(labels=['PROCTYP','PROC1'],axis=1)
        df = df.drop_duplicates()
        data = pd.concat([data,df], axis=0)
        
    data = data.replace({'DXVER': np.nan}, {'DXVER': '9'})
    data.SVCDATE = pd.to_datetime(data.SVCDATE)
    data.sort_values(by=['ENROLID', 'SVCDATE'], inplace=True)
    data = data.reset_index(drop=True)
    
    return data
    
    

def icd9_to_icd10(dx, mapping):
    if '.' in dx:
        dx = dx[:-2]  
    
    if dx in mapping.keys():
        dx = dx
    elif '0'+dx in mapping.keys():
        dx = '0'+dx 
    elif '00'+dx in mapping.keys():
        dx = '00'+dx
    elif '000'+dx in mapping.keys():
        dx = '000'+dx
    else:
        print(dx,'not in dict')
        return None
        
    return mapping[dx]

def map_function(x_dxs, mapping):
    map_list = list()
    for x_dx in x_dxs:
        mapped = icd9_to_icd10(x_dx, mapping)
        if mapped is not None:
            map_list.append(mapped)
            
    return map_list

def get_dx_list(patient_x, X_DXVER, mapping, DX_col):
    X = list()
    for date in patient_x.SVCDATE.unique():
        record = patient_x[patient_x.SVCDATE == date]
        if len(record) == 1:
            x_dxs = sum(record[DX_col].dropna(axis=1).values.tolist(), [])

            if X_DXVER == '0':
                X.append(x_dxs)
            else:
                x_dxs_mapped = map_function(x_dxs, mapping)
                if len(x_dxs_mapped) != 0:
                    X.append(x_dxs_mapped)
        else:
            x_dxs = sum(record[DX_col].values.tolist(), []) 
            x_dxs = list(map(str, list(set(x_dxs))))
            x_dxs = [x for x in x_dxs if x != 'nan']

            if X_DXVER == '0':
                X.append(x_dxs)
            else:
                x_dxs_mapped = map_function(x_dxs, mapping)
                if len(x_dxs_mapped) != 0:
                    X.append(x_dxs_mapped)    
                    
    return X
    
def check_target(dx, target_dict):
    for key in target_dict.keys():
        for code in target_dict[key]:
            if dx.startswith(code):
                return True, key
    return False, None
    

def get_cases(df, batch_ids, target_dict, mapping, config):
    n_visits = config.n_visits

    DX_col = [col for col in df.columns if 'DX' in col and col != 'DXVER']
    
    case_group = {
        'ID':[],
        'DXVER': [],
        'X': [],
        'Y': [],
        'X_idx': [],
        'Y_idx': []
    }
    control_group_ids = []        
    for id_ in tqdm(batch_ids, total=len(batch_ids)):
        is_saved = False
        patient = df[df.ENROLID == id_].reset_index(drop=True)
        if len(patient) < n_visits:
            continue
        if (patient.SVCDATE[-1:].item()-patient.SVCDATE[n_visits-1]).days < 365:
            continue
            
        for i, (_, row) in enumerate(patient.iterrows()):
            DXVER = '9'
            if not pd.isnull(row['DXVER']):
                DXVER = row['DXVER']
    
            dxs = list(row[DX_col])
            for dx in dxs:
                if not pd.isnull(dx):
                    is_case, target = check_target(dx, target_dict[DXVER])
    
            if is_case and i < n_visits:
                break
            if is_case and i >= n_visits:
                if (row['SVCDATE']-patient.SVCDATE[n_visits-1]).days < 365:
                    break
                for j in range(i-1, n_visits-2, -1):
                    if (row['SVCDATE']-patient.SVCDATE[j]).days > 365:
                        patient_ = patient.iloc[:j+1,:]
                        for X_DXVER in ['9','0']:
                            patient_x = patient_[patient_.DXVER==X_DXVER]
                            if len(patient_x.SVCDATE.unique()) >= n_visits:
                                X = get_dx_list(patient_x, X_DXVER, mapping, DX_col)
                                if len(X) >= n_visits:
                                    is_saved = True
                                    Y = target
                                    X_idx = j
                                    Y_idx = i
                                    case_group['ID'].append(id_)
                                    case_group['DXVER'].append(X_DXVER)
                                    case_group['X'].append(X)
                                    case_group['Y'].append(Y)
                                    case_group['X_idx'].append(X_idx)
                                    case_group['Y_idx'].append(Y_idx)

                    if is_saved:
                        break
                break
    

        if not is_case:
            control_group_ids.append(id_)
            
        
    np.save(os.path.join(config.save_dir, 'control_group', control_group_ids)  
    np.save(os.path.join(config.save_dir, 'case_group', case_group) 
    
    

        
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--path', default = './data/',type=str)
    args.add_argument('--save_dir', default = './data_npy/',type=str)
    args.add_argument('--n_visits', default = 10,type=int)
    
    config = args.parse_args()
    
    target_dict = {
        '9': {
            'HF': ['425', '428', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', 'K77'],
            'AF':['42731'],
            'CAD':['4111','41181','41189','4130','4131','4139','412','41400','41401','4142','4143','4148','4149',
                        '4292','41402','41403','41404','41405','V4581','V4582'],     ###
            'PAD':['4439','4438','44020','44021'],
            'ST': ['4380', '4381', '4382', '4383', '4384', '4385', '4386', '4387', '4388', '4389', 'V1254'],
            'HT': ['401','403','405']  ###
            
        },
        '0':{
            'HF': ['I11', 'I13', 'I50', 'I42', 'K77'],
            'AF': ['I480','I481','I482','I4891'],
            'CAD': ['I20', 'I240', 'I248','I25'],
            'PAD': ['I73'],
            'ST': ['Z8673', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'G458', 'G459'],
            'HT': ['I10','I12','I15','I16'] ###
        }
    }
    
    mapping = pd.read_csv('icd9to10.csv')[['icd9cm', 'icd10cm']]
    mapping = dict(mapping.values.tolist())
        
    df = load_dataset(config.path)

    all_ids = df.ENROLID.unique()

    print(f'all unique id: {len(all_ids)}')
    
    get_cases(df, all_ids, target_dict, mapping, config)
 
