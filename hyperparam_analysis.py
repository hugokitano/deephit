import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import seaborn as sns

def read_file(hyperparameter_file):
    data = pd.read_csv(hyperparameter_file, header=None) 
    #pdb.set_trace()
    for index, row in data.iterrows():
        row.iloc[0] = row.iloc[0].split(":")[1]  #mb_size
        row.iloc[1] = row.iloc[1].split(":")[1] #h_dim_shared 
        row.iloc[2] = row.iloc[2].split(":")[1] #h_dim_CS
        row.iloc[3] = row.iloc[3].split(":")[1] #num_layers_shared
        row.iloc[4] = row.iloc[4].split(":")[1] #num_layers_CS
        row.iloc[5] = row.iloc[5].split(":")[1] # active_fn
        row.iloc[6] = row.iloc[6].split(":")[1] # beta
        data.iloc[index] = row
    data.columns = ['mb_size', 'nodes_shared', 'nodes_CS', 'num_layers_shared', 'num_layers_CS', 'activation_fn', 'beta', 'c_index', 'brier_index']
    data.mb_size = data.mb_size.astype('category').cat.reorder_categories(['32', '64', '128'])
    data.nodes_shared = data.nodes_shared.astype('category').cat.reorder_categories(['25', '50', '75', '100'])
    data.nodes_CS = data.nodes_CS.astype('category').cat.reorder_categories(['25', '50', '75', '100'])
    data.num_layers_shared = data.num_layers_shared.astype('category')
    data.num_layers_CS = data.num_layers_CS.astype('category')
    data.activation_fn = data.activation_fn.astype('category')
    data.beta = data.beta.astype('category')
    data.c_index = data.c_index.astype(float)
    data.brier_index = data.brier_index.astype(float)
    return data

def plot_data(data):
    for i in range(7):
        column_name =data.iloc[:,i].name
        sns_plot = sns.catplot(x=column_name, y="c_index", kind="box", data=data);
        sns_plot.savefig('plots/' + 'MEC_' + column_name + '.png')



def main():
    hyperparameter_file = "MEC_constrained/hyperparameters.csv"
    data = read_file(hyperparameter_file)
    plot_data(data)

if __name__ == '__main__':
    main()