_EPSILON = 1e-08

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import pdb
import sys

from termcolor import colored
from tensorflow.contrib.layers import fully_connected as FC_Net
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

import import_data as impt
import utils_network as utils

from class_DeepHit import Model_DeepHit
from utils_eval import c_index, brier_score, weighted_c_index, weighted_brier_score


def load_logging(filename):
    data = dict()
    with open(filename) as f:
        def is_float(input):
            try:
                num = float(input)
            except ValueError:
                return False
            return True

        for line in f.readlines():
            if ':' in line:
                key,value = line.strip().split(':', 1)
                if value.isdigit():
                    data[key] = int(value)
                elif is_float(value):
                    data[key] = float(value)
                elif value == 'None':
                    data[key] = None
                else:
                    data[key] = value
            else:
                pass # deal with bad lines of text here    
    return data



##### MAIN SETTING
OUT_ITERATION               = 5

data_mode                   = 'MEC' #METABRIC, SYNTHETIC
seed                        = 1234

EVAL_TIMES                  = [5, 10, 15] # evalution times (for C-index and Brier-Score)


##### IMPORT DATASET
'''
    num_Category            = max event/censoring time * 1.2 (to make enough time horizon)
    num_Event               = number of evetns i.e. len(np.unique(label))-1
    max_length              = maximum number of measurements
    x_dim                   = data dimension including delta (num_features)
    mask1, mask2            = used for cause-specific network (FCNet structure)
'''
pdb.set_trace()

if len(sys.argv) < 2:
    print("please specify dataset and out directory")
in_filename = sys.argv[1]
outpath = sys.argv[2]


(x_dim), (data, time, label), (mask1, mask2) = impt.import_dataset_MEC(norm_mode = 'standard')
(te_x_dim), (te_data, te_time, te_label), (te_mask1, te_mask2), (te_ids, te_data_raw) = impt.import_dataset_general(in_filename, norm_mode = 'standard')

EVAL_TIMES = [5, 10, 15]

_, num_Event, num_Category  = np.shape(mask1)  # dim of mask1: [subj, Num_Event, Num_Category]
    


in_path = 'MEC/results/'

if not os.path.exists(in_path):
    os.makedirs(in_path)


FINAL1 = np.zeros([num_Event, len(EVAL_TIMES), OUT_ITERATION])
FINAL2 = np.zeros([num_Event, len(EVAL_TIMES), OUT_ITERATION])

pdb.set_trace()
for out_itr in range(OUT_ITERATION):
    in_hypfile = in_path + 'itr_' + str(out_itr) + '/hyperparameters_log.txt'
    in_parser = load_logging(in_hypfile)


    ##### HYPER-PARAMETERS
    mb_size                     = in_parser['mb_size']

    iteration                   = in_parser['iteration']

    keep_prob                   = in_parser['keep_prob']
    lr_train                    = in_parser['lr_train']

    h_dim_shared                = in_parser['h_dim_shared']
    h_dim_CS                    = in_parser['h_dim_CS']
    num_layers_shared           = in_parser['num_layers_shared']
    num_layers_CS               = in_parser['num_layers_CS']

    if in_parser['active_fn'] == 'relu':
        active_fn                = tf.nn.relu
    elif in_parser['active_fn'] == 'elu':
        active_fn                = tf.nn.elu
    elif in_parser['active_fn'] == 'tanh':
        active_fn                = tf.nn.tanh
    else:
        print('Error!')


    initial_W                   = tf.contrib.layers.xavier_initializer()

    alpha                       = in_parser['alpha']  #for log-likelihood loss
    beta                        = in_parser['beta']  #for ranking loss
    gamma                       = in_parser['gamma']  #for RNN-prediction loss
    parameter_name              = 'a' + str('%02.0f' %(10*alpha)) + 'b' + str('%02.0f' %(10*beta)) + 'c' + str('%02.0f' %(10*gamma))


    ##### MAKE DICTIONARIES
    # INPUT DIMENSIONS
    input_dims                  = { 'x_dim'         : x_dim,
                                    'num_Event'     : num_Event,
                                    'num_Category'  : num_Category}

    # NETWORK HYPER-PARMETERS
    network_settings            = { 'h_dim_shared'         : h_dim_shared,
                                    'h_dim_CS'          : h_dim_CS,
                                    'num_layers_shared'    : num_layers_shared,
                                    'num_layers_CS'    : num_layers_CS,
                                    'active_fn'      : active_fn,
                                    'initial_W'         : initial_W }


    # for out_itr in range(OUT_ITERATION):
    print ('ITR: ' + str(out_itr+1) + ' DATA MODE: ' + data_mode + ' (a:' + str(alpha) + ' b:' + str(beta) + ' c:' + str(gamma) + ')' )
    ##### CREATE DEEPFHT NETWORK
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)
    saver = tf.train.Saver()

    # NEW CODE
    writer = tf.summary.FileWriter(in_path + '/itr_' + str(out_itr) + '/graphs', sess.graph)
    
    sess.run(tf.global_variables_initializer())

    ### TRAINING-TESTING SPLIT
    (tr_data, va_data, tr_time, va_time, tr_label, va_label, 
     tr_mask1,va_mask1, tr_mask2, va_mask2)  = train_test_split(data, time, label, mask1, mask2, test_size=0.20, random_state=seed) 
    
    ##### PREDICTION & EVALUATION
    saver.restore(sess, in_path + '/itr_' + str(out_itr) + '/models/model_itr_' + str(out_itr))

    ### PREDICTION
    # pdb.set_trace()
    pred = model.predict(te_data)
    


    #test_id + test data + test_label + test_time + test_pred
    #1       + 29          1          + 1         + 28

    test_id = pd.DataFrame(te_ids).astype(int)
    test_data_raw = pd.DataFrame(te_data_raw)
    test_label = pd.DataFrame(te_label).astype(int)
    test_time = pd.DataFrame(te_time).astype(int)
    test_pred = pd.DataFrame(pred[:,0,:])  

    test_predictions = pd.concat([test_id, test_data_raw, test_label, test_time, test_pred], axis = 1)  
    data_length = test_data_raw.shape[1]
    data_columns = ["data_" + str(i) for i in range(data_length)]
    pred_length = test_pred.shape[1]
    pred_columns = ["prob" + str(i) for i in range(pred_length)]
    id_column = ["id"]
    time_column = ["Time"]
    event_column = ['event']
    columns = id_column + data_columns + time_column + event_column + pred_columns

    test_predictions.columns = columns
    test_predictions.to_csv(outpath + '/predictions_' + str(out_itr) + '.csv')

    
    ### EVALUATION
    result1, result2 = np.zeros([num_Event, len(EVAL_TIMES)]), np.zeros([num_Event, len(EVAL_TIMES)])

    # pdb.set_trace()
    for t, t_time in enumerate(EVAL_TIMES):
        eval_horizon = int(t_time)

        if eval_horizon >= num_Category:
            print( 'ERROR: evaluation horizon is out of range')
            result1[:, t] = result2[:, t] = -1
        else:
            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            #0 to eval_horizion 
            risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until EVAL_TIMES
            for k in range(num_Event):
                # result1[k, t] = c_index(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                # result2[k, t] = brier_score(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)

                #train time, train labels for cause k, risk, test time, test labels for cause k, eval_horizon
                result1[k, t] = weighted_c_index(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                result2[k, t] = weighted_brier_score(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)

    FINAL1[:, :, out_itr] = result1
    FINAL2[:, :, out_itr] = result2

    ### SAVE RESULTS
    row_header = []
    for t in range(num_Event):
        row_header.append('Event_' + str(t+1))

    col_header1 = []
    col_header2 = []
    for t in EVAL_TIMES:
        col_header1.append(str(t) + 'yr c_index')
        col_header2.append(str(t) + 'yr B_score')

    # c-index result
    df1 = pd.DataFrame(result1, index = row_header, columns=col_header1)
    df1.to_csv(outpath + '/result_CINDEX_itr' + str(out_itr) + '.csv')

    # brier-score result
    df2 = pd.DataFrame(result2, index = row_header, columns=col_header2)
    df2.to_csv(outpath + '/result_BRIER_itr' + str(out_itr) + '.csv')

    ### PRINT RESULTS
    print('========================================================')
    print('ITR: ' + str(out_itr+1) + ' DATA MODE: ' + data_mode + ' (a:' + str(alpha) + ' b:' + str(beta) + ' c:' + str(gamma) + ')' )
    print('SharedNet Parameters: ' + 'h_dim_shared = '+str(h_dim_shared) + ' num_layers_shared = '+str(num_layers_shared) + 'Non-Linearity: ' + str(active_fn))
    print('CSNet Parameters: ' + 'h_dim_CS = '+str(h_dim_CS) + ' num_layers_CS = '+str(num_layers_CS) + 'Non-Linearity: ' + str(active_fn)) 

    print('--------------------------------------------------------')
    print('- C-INDEX: ')
    print(df1)
    print('--------------------------------------------------------')
    print('- BRIER-SCORE: ')
    print(df2)
    print('========================================================')


    
### FINAL MEAN/STD
# c-index result
df1_mean = pd.DataFrame(np.mean(FINAL1, axis=2), index = row_header, columns=col_header1)
df1_std  = pd.DataFrame(np.std(FINAL1, axis=2), index = row_header, columns=col_header1)
df1_mean.to_csv(outpath + '/result_CINDEX_FINAL_MEAN.csv')
df1_std.to_csv(outpath + '/result_CINDEX_FINAL_STD.csv')

# brier-score result
df2_mean = pd.DataFrame(np.mean(FINAL2, axis=2), index = row_header, columns=col_header2)
df2_std  = pd.DataFrame(np.std(FINAL2, axis=2), index = row_header, columns=col_header2)
df2_mean.to_csv(outpath + '/result_BRIER_FINAL_MEAN.csv')
df2_std.to_csv(outpath + '/result_BRIER_FINAL_STD.csv')


### PRINT RESULTS
print('========================================================')
print('- FINAL C-INDEX: ')
print(df1_mean)
print('--------------------------------------------------------')
print('- FINAL BRIER-SCORE: ')
print(df2_mean)
print('========================================================')