import numpy as np
import pandas as pd
import pdb

def read_files():
    pdb.set_trace()

    in_filename = 'total.csv'
    df = pd.read_csv(in_filename, sep =',') # (5743, 33)



    df = df.sort_values(by=['id'])
    df = df.drop(df.columns[0],axis=1)
    df.to_csv('total2.csv')







def f_get_Normalization(X, norm_mode):
    num_Patient, num_Feature = np.shape(X)

    if norm_mode == 'standard': #zero mean unit variance
        for j in range(num_Feature):
            if np.std(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))/np.std(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))
    elif norm_mode == 'normal': #min-max normalization
        for j in range(num_Feature):
            X[:,j] = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j]) - np.min(X[:,j]))
    else:
        print("INPUT MODE ERROR!")

    return X

def main():
    read_files()

if __name__ == '__main__':
    main()