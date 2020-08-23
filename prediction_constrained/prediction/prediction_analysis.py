import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import seaborn as sns

def read_files():
    pdb.set_trace()
    train_data = np.loadtxt('train_data.txt', delimiter = ",") #1149 x 29
    train_label = np.loadtxt('train_event.txt', delimiter = ",") #1149
    train_time = np.loadtxt('train_time.txt', delimiter = ",") #1149
    train_pred = np.loadtxt('train_pred.txt', delimiter = ",") #1149 x 28

    val_data = np.loadtxt('val_data.txt', delimiter = ",") #1149 x 29
    val_label = np.loadtxt('val_event.txt', delimiter = ",") #1149
    val_time = np.loadtxt('val_time.txt', delimiter = ",") #1149
    val_pred = np.loadtxt('val.txt', delimiter = ",") #1149 x 28

    test_data = np.loadtxt('test_data.txt', delimiter = ",") #1149 x 29
    test_label = np.loadtxt('test_event.txt', delimiter = ",") #1149
    test_time = np.loadtxt('test_time.txt', delimiter = ",") #1149
    test_pred = np.loadtxt('test_pred.txt', delimiter = ",") #1149 x 28

    in_filename = '../sample data/MEC/MEC_cleaned_SPLC_years.csv'
    df = pd.read_csv(in_filename, sep =',')

    label = np.asarray(df[['event']]) 
    time = np.asarray(df[['Time']])

    data = np.asarray(df.iloc[:,2:-2])
    ids = np.asarray(df.iloc[:,0])
    ids = ids.reshape(ids.shape[0], 1)
    data_raw  = np.copy(data) 
    data = f_get_Normalization(data, 'standard')
    data = np.hstack((ids,data))


    #train
    train_id = np.zeros(train_data.shape[0])
    test_index = np.zeros(train_data.shape[0])
    train_data_raw = np.zeros_like(train_data)
    for i in range(0, train_data.shape[0]):
        for j in range(0, data.shape[0]):
            if np.allclose(train_data[i,:], data[j,1:]):
                train_id[i] = data[j,0]
                train_index[i] = j
                train_data_raw[i,:] = data_raw[j,:]
    pdb.set_trace()
    train_id = train_id.astype(int)
    train_index = train_index.astype(int)
    is_train = np.ones(train_data.shape[0])


    train_id = pd.DataFrame(train_id).astype(int)
    is_train = pd.DataFrame(is_train).astype(int)
    train_data_raw = pd.DataFrame(train_data_raw)
    train_label = pd.DataFrame(train_label).astype(int)
    train_time = pd.DataFrame(train_time).astype(int)
    train_pred = pd.DataFrame(train_pred) 
    train = pd.concat([train_id, is_train, train_data_raw, train_label, train_time, train_pred], axis = 1)     

    #val
    val_id = np.zeros(val_data.shape[0])
    val_index = np.zeros(val_data.shape[0])
    val_data_raw = np.zeros_like(val_data)
    for i in range(0, val_data.shape[0]):
        for j in range(0, data.shape[0]):
            if np.allclose(val_data[i,:], data[j,1:]):
                val_id[i] = data[j,0]
                val_index[i] = j
                val_data_raw[i,:] = data_raw[j,:]
    pdb.set_trace()
    val_id = val_id.astype(int)
    val_index = val_index.astype(int)
    is_train = np.ones(val_data.shape[0])


    val_id = pd.DataFrame(val_id).astype(int)
    is_train = pd.DataFrame(is_train).astype(int)
    val_data_raw = pd.DataFrame(val_data_raw)
    val_label = pd.DataFrame(val_label).astype(int)
    val_time = pd.DataFrame(val_time).astype(int)
    val_pred = pd.DataFrame(val_pred)   
    val = pd.concat([val_id, is_train, val_data_raw, val_label, val_time, val_pred], axis = 1)    


    test_id = np.zeros(test_data.shape[0])
    test_index = np.zeros(test_data.shape[0])
    test_data_raw = np.zeros_like(test_data)
    for i in range(0, test_data.shape[0]):
        for j in range(0, data.shape[0]):
            if np.allclose(test_data[i,:], data[j,1:]):
                test_id[i] = data[j,0]
                test_index[i] = j
                test_data_raw[i,:] = data_raw[j,:]
    pdb.set_trace()
    test_id = test_id.astype(int)
    test_index = test_index.astype(int)
    is_train = np.zeros(test_data.shape[0])
    #test_id + test data + test_label + test_time + prediction
    #1       + 29        + 1            1         + 28

    test_id = pd.DataFrame(test_id).astype(int)
    is_train = pd.DataFrame(is_train).astype(int)
    test_data_raw = pd.DataFrame(test_data_raw)
    test_label = pd.DataFrame(test_label).astype(int)
    test_time = pd.DataFrame(test_time).astype(int)
    test_pred = pd.DataFrame(test_pred)      

    test = pd.concat([test_id, test_data_raw, test_label, test_time, prediction], axis = 1)  
    tofile = pd.concat([train, val, test])

    tofile.columns = ["id", "is_train", "age_ix","packyears2","cigday2","quityears2","bmi","sex","race.catA","race.catB","race.catH",
        "race.catHW","race.catO","edu1","edu3","hist3.ixLC","hist3.ixOTH","hist3.ixSC","hist3.ixSQ","smkstatus23","USPSTF","synchronous.ix",
        "copd1","copd9","ph","fh","stage2.ix1","surgery_ix1","surgery_ix9","radiation_ix1","radiation_ix9","Time","event", "prob_0", "prob_1", 
        "prob_2",  "prob_3",  "prob_4",  "prob_5", "prob_6", "prob_7", "prob_8", "prob_9",  "prob_10", "prob_11",  "prob_12", "prob_13",  "prob_14",
        "prob_15",  "prob_16",  "prob_17",  "prob_18", "prob_19", "prob_20", "prob_21",  "prob_22",  "prob_23",  "prob_24",  "prob_25",  "prob_26",
        "prob_27"]

    tofile.sort_values(by=['is_train'])
    tofile.to_csv('total.csv')







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