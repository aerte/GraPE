from grape.datasets import *
import numpy as np
from grape.analysis import classyfire, classyfire_result_analysis
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import pickle
from grape.utils import DataSet

plt.rcParams.update(
    {
        "text.usetex": True
    }
)


seed = 12
np.random.seed(seed)

# Set dataset
# Decide if the ids were already retrieved
already_run = False

if not already_run:

    #dataset = DataSet(file_path='./data/processed/QM9.pickle')
    dataset = BradleyDoublePlus()

    data = dataset.smiles
    ids = np.random.choice(len(data), size=500, replace=False)
    id_smiles, id_relative = classyfire(data[ids])


    mol_dic, freq_dic = classyfire_result_analysis(idx=id_relative, layer=1)
    with open('./results/bradley.pkl', 'wb') as f:
        pickle.dump(freq_dic, f)

# rcParams.update({'figure.autolayout': True})
# freq_dic = pickle.load(open('./results/free1.pkl', 'rb'))
# x = np.array(list(freq_dic.keys()))
# # Sort keys for consistency
# x = np.sort(x)
#
# y = freq_dic.values()
#
# plt.rcParams.update({'font.size': 26})
#
# fig, ax = plt.subplots(figsize=(11,10))
# palette = sns.color_palette('muted', n_colors=len(freq_dic.keys()))
#
#
# sns.barplot(ax=ax,x=y, y=x, orient='h', palette=palette)
# ax.set_title('FreeSolv \n compound class frequencies ')
# ax.set_xlabel(r'Frequency \\\\ $\textbf{(c)}$')
# ax.set_xticks(range(0,250,50))
# fig.savefig('./results/plots/free1.png')
#
# plt.show()