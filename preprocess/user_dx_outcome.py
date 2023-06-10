from collections import defaultdict
from datetime import datetime
import os
from tqdm import tqdm
import pandas as pd
import pickle


def my_dump(obj, file_name):
    pickle.dump(obj, open(file_name, 'wb'))
    print(f'dumped {file_name}...', flush=True)
    
    
def is_valid_target(icd_code, target_list):
    for code in target_list:
        if icd_code.startswith(code):
            return True
    return False


def get_code_list(icd_codes, DXVER, mapping_dict, target_dict):
    saved_codes = list()
    valid_target = list()
    for icd_code in icd_codes:
        if not pd.isnull(icd_code):
            if DXVER == '9':
                for i in range(3):
                    if icd_code+'0'*i in mapping_dict:
                        mapped = mapping_dict.get(icd_code+'0'*i)
                        saved_codes.append(mapped[:3])
                        for target, target_list in target_dict.items():
                            if is_valid_target(mapped, target_list):
                                valid_target.append(target)
                        break
            else:
                saved_codes.append(icd_code[:3])
                for target, target_list in target_dict.items():
                    if is_valid_target(icd_code, target_list):
                        valid_target.append(target)

    return saved_codes, valid_target


def get_user_dx_outcome(args, mapping_dict, target_dict):
    user_dx = defaultdict(dict)
    case_cohort = dict()
    for target in target_dict.keys():
        case_cohort[target] = defaultdict(list)
    
    files = os.listdir(args.input_dir)
    for file in files:
        if 'inpat' in file or 'outpat' in file :
            input_file = os.path.join(args.input_dir, file)
            inpat = pd.read_csv(input_file, dtype=str)
            print(f'load {input_file}...', flush=True)

            DATE_NAME = [col for col in inpat.columns if 'DATE' in col][0]
            DX_col = [col for col in inpat.columns if 'DX' in col and col != 'DXVER']

            DXVER_col = False            
            if 'DXVER' in inpat.columns:
                DXVER_col = True

            inpat = inpat[~inpat[DATE_NAME].isnull()]
            inpat = inpat[~inpat['DX1'].isnull()]

            for index, row in tqdm(inpat.iterrows(), total=len(inpat)):
                dxs = list(row[DX_col])
                enrolid = row['ENROLID']
                date = row[DATE_NAME]
                date = datetime.strptime(str(date), '%m/%d/%Y')
                DXVER = '9'
                if DXVER_col:  # files after 2015: a mix usage of both ICD-9 codes andd ICD-10 codes;
                    DXVER = row['DXVER']
                if DXVER == '':
                    continue

                dxs, valid_target = get_code_list(dxs, DXVER, mapping_dict, target_dict)  

                for target in list(set(valid_target)):
                    case_cohort[target][enrolid].append(date)

                if enrolid not in user_dx:
                    user_dx[enrolid][date] = dxs
                else:
                    if date not in user_dx[enrolid]:
                        user_dx[enrolid][date] = dxs
                    else:
                        user_dx[enrolid][date].extend(dxs)

    print('Preprocessing completed..', flush=True)
               
    my_dump(user_dx, os.path.join(args.pkl_dir, 'user_dx.pkl'))
    my_dump(case_cohort, os.path.join(args.pkl_dir, 'case_cohort.pkl'))
    
    print(f'> length of user_dx: {len(user_dx)}', flush=True)
    for target in target_dict.keys():
        print(f'> length of case_cohort_{target}: {len(case_cohort[target])}', flush=True)
        
    return user_dx, case_cohort
