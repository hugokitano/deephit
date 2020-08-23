import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import seaborn as sns

def read_files(deephit_file, fgr_file):
    deephit = pd.read_csv(deephit_file) 
    fgr = pd.read_csv(fgr_file)
    deephit_risk = []
    fgr_risk = []
    for index, fgr_row in fgr.iterrows():
        patient_id = fgr_row['Obs']
        deephit_row = deephit.loc[deephit['id'] == patient_id]
        if len(deephit_row) == 0: # no patient in deephit
            continue
        # if fgr_row['cigday2'] != deephit_row['cigday2'][0] or fgr_row['event'] != deephit_row['event'][0]:
        #     print('patient id: ' + str(patient_id) + ' is mismatched')
        #     continue
        fgr_5_risk = fgr_row['FGR-time5']
        deephit_5_risk = deephit_row.iloc[0,34:39].sum()
        fgr_risk.append(fgr_5_risk)
        deephit_risk.append(deephit_5_risk)
    return fgr_risk, deephit_risk

def plot_data(data):
    fgr_risk, deephit_risk = data
    plt.scatter(fgr_risk, deephit_risk)
    plt.title('5-year risk comparison')
    plt.xlabel('FGR')
    plt.ylabel('DeepHit')
    plt.show()



def main():
    deephit_file = "prediction_constrained/total.csv"
    fgr_file = "mec-shiny-results.csv"
    data = read_files(deephit_file, fgr_file)
    plot_data(data)

if __name__ == '__main__':
    main()