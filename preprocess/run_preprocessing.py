import os
import pandas as pd
import argparse
import pickle
from user_dx_outcome import get_user_dx_outcome
from save_cohort import save_input_cohort
from save_case_control_cohort import select_cohort


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--input_dir', default='data', type=str, help='input data directory')
    parser.add_argument('--pkl_dir', default='pickles', type=str)
    parser.add_argument('--demo', default='data/demo.csv', type=str)
    parser.add_argument('--mapping', default='preprocess/icd9to10.csv', type=str)
    parser.add_argument('--pred_windows', default=[90, 180, 360])
    parser.add_argument('--min_visits', default=10, type=int)
    args = parser.parse_args()
        
    if not os.path.exists(args.pkl_dir):
        os.makedirs(args.pkl_dir)
        
    mapping_dict = pd.read_csv(args.mapping)[['icd9cm', 'icd10cm']]
    mapping_dict = dict(mapping_dict.values.tolist())
    demo = pd.read_csv(args.demo)
    
    target_dict =   {
        'HF': ['I11', 'I13', 'I50', 'I42', 'K77'],
        'ST': ['Z8673', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'G458', 'G459']
    }
    print('Save user dx and outcome...', flush=True)
    user_dx, case_cohort = get_user_dx_outcome(args, mapping_dict, target_dict)
    print(f'Preprocess input cohort...', flush=True)
    save_input_cohort(args, user_dx, case_cohort, demo)
    
    
    for target in target_dict.keys():
        for window in args.pred_windows:
            print(f'Save case and control cohort for {target} with {window} window...', flush=True)
            data = pickle.load(open(os.path.join(
                                args.pkl_dir, f'{target}_saved_cohort_{window}.pkl'
                            ), 'rb'))
            data = pd.DataFrame(data)

            cases = data[data['Y'] == target]
            controls = data[data['Y'] == 'Normal']

            cohort = select_cohort(cases, controls)
            pickle.dump(cohort.to_dict('list'), open(
                    os.path.join(args.pkl_dir, 'input_{}_{}.pkl'.format(target, window)) , 'wb'
                    ))
