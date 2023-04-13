import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os 
import copy


def load_raw(path='./data_market/'):
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
    elif dx+'0' in mapping.keys():
        dx = dx+'0'
    elif '00'+dx in mapping.keys():
        dx = '00'+dx
    elif '000'+dx in mapping.keys():
        dx = '000'+dx
    else:
        print(dx,'not in dict')
        return None

    return mapping[dx]

def check_target(dx, target_list, DXVER, mapping):
    if DXVER == '9':
        dx = icd9_to_icd10(dx, mapping)
        if dx is None:
            return False
    for code in target_list:
        if dx.startswith(code):
            return True
    return False
    
    
def get_code_list(patient_x, mapping, DX_col):
    DX, DX_f = list(), list()
    DATE = list()

    for date_ in patient_x.SVCDATE.unique():
        records = patient_x[patient_x.SVCDATE == date_]
        x_dxs, x_dxs_f = list(), list()
        for _, row in records.iterrows():
            DXVER = row['DXVER']
            dxs = list(row[DX_col])
            for dx in dxs:
                if not pd.isnull(dx):
                    if DXVER == '9':
                        mapped = icd9_to_icd10(dx, mapping)
                        if mapped is not None:
                            x_dxs.append(mapped[:3])
                            x_dxs_f.append(mapped)
                    else:
                        x_dxs.append(dx[:3])
                        x_dxs_f.append(dx)
            
        x_dxs = list(set(x_dxs))
        x_dxs_f = list(set(x_dxs_f))

        DX.append(x_dxs)
        DX_f.append(x_dxs_f)
        DATE.append(pd.to_datetime(date_).strftime('%Y-%m'))
            
    return DX, DX_f, DATE

class SearchData:
    def __init__(self, df, target_list, mapping, demo, config, windows=[90, 180, 360], DXVER_list=['0','9']):
        super(SearchData, self).__init__()
        self.target_list = target_list
        self.target = config.target
        self.mapping = mapping
        self.demo = demo
        self.config = config
        self.n_visits = config.n_visits
        self.DX_col = [col for col in df.columns if 'DX' in col and col != 'DXVER']

   
        self.windows = windows
        self.DXVER_list = DXVER_list
        self.dict_ = dict()
        for window in self.windows:
            self.dict_[window] = {
                'ID':[],
                'DX': [],
                'DX_f': [],
                'DATE': [],
                'Y': [],
                'n_visit': [],
                'last_observed_date': [],
                'year': [],
                'month': [],
                'target_date':[],
                'age': [],
                'gender':[]
            }

        
    def add_data(self, dict_, patient_x, SVCDATE, id_, patient_demo, target):
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
            DX_list, DX_f_list, DATE_list = get_code_list(patient_x, self.mapping, self.DX_col)
            for window, date in save_date.items():
                if date['date_idx'] == 0:
                    DX = DX_list
                    DX_f = DX_f_list
                    DATE = DATE_list
                else:
                    DX = DX_list[:-date['date_idx']]
                    DX_f = DX_f_list[:-date['date_idx']]
                    DATE = DATE_list[:-date['date_idx']]
                    
                dict_[window]['ID'].append(id_)
                dict_[window]['DX'].append(DX)
                dict_[window]['DX_f'].append(DX_f)
                dict_[window]['DATE'].append(DATE)
                dict_[window]['Y'].append(target)
                dict_[window]['n_visit'].append(len(DX))
                dict_[window]['last_observed_date'].append(date['last_observed_date'])
                dict_[window]['year'].append(date['last_observed_date'].strftime('%Y'))
                dict_[window]['month'].append(date['last_observed_date'].strftime('%m'))
                dict_[window]['target_date'].append(pd.to_datetime(SVCDATE))
                dict_[window]['age'].append(patient_demo['DOBYR'].item())
                dict_[window]['gender'].append(patient_demo['SEX'].item())
                
        return dict_
    
    def forward(self, ids):
        case_group = copy.deepcopy(self.dict_)
        control_group = copy.deepcopy(self.dict_)
        control_ids = []
        for id_ in tqdm(ids, total=len(ids)):
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
                        is_case = check_target(dx, self.target_list, DXVER, self.mapping)
                        if is_case:
                            break                    
                if is_case and i < self.n_visits:
                    break
                if is_case and i >= self.n_visits:
                    patient_x = patient.iloc[:i,:]
                    case_group = self.add_data(case_group, patient_x, SVCDATE, id_, patient_demo, self.target)
                    break
                    
            if not is_case:
                SVCDATE= patient.SVCDATE.tolist()[-1]
                patient_x = patient.iloc[:-1,:]   
                control_group = self.add_data(control_group, patient_x, SVCDATE, id_, patient_demo, 'normal')
                
                         
        for window in self.windows:  
            save_dir = './data_npy/{}/{}'.format(self.target, window)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  
            np.save(os.path.join(save_dir, 'case_group_{}'.format(self.config.fold_id)), case_group[window]) 
            np.save(os.path.join(save_dir, 'control_group_{}'.format(self.config.fold_id)), control_group[window]) 



        
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--path', default = './data_market/',type=str)
    args.add_argument('--n_visits', default = 10, type=int)
    args.add_argument('--target', type=str)
    args.add_argument('--n_split', type=int)
    args.add_argument('--fold_id', type=int)
    
    config = args.parse_args()
    
    if config.target == 'hf':
        target_list = ['I11', 'I13', 'I50', 'I42', 'K77']
    
    else:
        target_list = ['Z8673', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'G458', 'G459']

    
    mapping = pd.read_csv('icd9to10.csv')[['icd9cm', 'icd10cm']]
    mapping = dict(mapping.values.tolist())
    
    df = load_raw(config.path)
    print(df)
    print('df_len:', len(df))


    all_ids = df.ENROLID.unique()
    batch_size = len(all_ids)//config.n_split
    
    demo = pd.read_csv('demo.csv', dtype=str)
    
    if config.fold_id == config.n_split-1:
        batch_ids= all_ids[config.fold_id*batch_size:]
    else:
        batch_ids= all_ids[config.fold_id*batch_size : (1+config.fold_id)*batch_size]
    
    print(f'all unique id: {len(all_ids)}, n split: {config.n_split}, batch size: {batch_size}, batch size: {len(batch_ids)}')
    
    search_data = SearchData(df, target_list, mapping, demo, config)
    search_data.forward(batch_ids)
    

 
