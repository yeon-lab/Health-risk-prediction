from tqdm import tqdm
import pandas as pd

def select_cohort(cases, controls):
    saved_case_idx = []
    saved_control_idx = []

    for case_idx, row in tqdm(cases.iterrows(), total = len(cases)):
        indices = controls[(controls.n_visit == row['n_visit']) & (controls.year == row['year']) & (controls.month == row['month']) &\
                                        (controls.age == row['age']) & (controls.gender == row['gender'])].index
        if len(indices) > 0:            
            for control_idx in indices:
                if control_idx in saved_control_idx:
                    continue
                else:
                    saved_control_idx.append(control_idx)
                    saved_case_idx.append(case_idx)
                    break
    cohort = pd.concat([cases.loc[saved_case_idx, ], controls.loc[saved_control_idx, ]])
    
    return cohort


