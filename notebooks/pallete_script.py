import numpy as np
from grape.analysis import classyfire, classyfire_result_analysis
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import pickle

plt.rcParams.update(
    {
        "text.usetex": True
    }
)

rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 30})


# freq_dic_qm9 = pickle.load(open('./results/qm9.pkl', 'rb'))
# freq_dic_logp = pickle.load(open('./results/logp.pkl', 'rb'))
# freq_dic_freesolv = pickle.load(open('./results/free1.pkl', 'rb'))
# freq_dic_bradley = pickle.load(open('./results/bradley.pkl', 'rb'))
#
# alls = list(freq_dic_qm9.keys()) + list(freq_dic_logp.keys()) + list(freq_dic_freesolv.keys())+list(freq_dic_bradley.keys())
#
# comp = list(set(alls))
#
#
# freq = freq_dic_freesolv
#
# for i in comp:
#     if i not in freq.keys():
#         freq[i] = 0
#
#
# y = np.array(list(freq.values()))
# x = np.array(list(freq.keys()))
#
# print(x)
# print(y)
#
# ids = np.argsort(x)
# y = y[ids]
# x = x[ids]
#
# print(y)
# print(x)
#
# # print(y)
#
# # x = np.array(list(freq_dic.keys()))
# # # Sort keys for consistency
# # ids = np.argsort(x)
# # x = x[ids]
#
# fig, ax = plt.subplots(figsize=(12,12))
# palette = sns.color_palette('muted', n_colors=len(x))
#
#
# sns.barplot(ax=ax,x=y, y=x, orient='h', palette=palette)
# title = 'FreeSolv \n Compound class frequencies'
# ax.set_title(title)
# ax.set_xlabel(r'Frequency \\\\ $\textbf{(d)}$')
# ax.set_xticks(range(0,200,50))
# fig.savefig('./results/plots/freesolv.png')
#
# plt.show()


freq_dic = pickle.load(open('./results/free0.pkl', 'rb'))
#print(freq_dic)
freq_dic['Inorganic compounds'] =  0
#print(freq_dic)

y = np.array(list(freq_dic.values()))
x = np.array(list(freq_dic.keys()))
ids = np.argsort(x)
y = y[ids]
x = x[ids]
# y = y[-15:]
# x = x[-15:]
# print(y)

# x = np.array(list(freq_dic.keys()))
# # Sort keys for consistency
# ids = np.argsort(x)
# x = x[ids]

fig, ax = plt.subplots(figsize=(12,14))
palette = sns.color_palette('muted', n_colors=len(freq_dic.keys()))


sns.barplot(ax=ax,x=y, y=x, orient='h', palette=palette)
ax.set_title('FreeSolv \n Compound class frequencies \n of the Kingdom Classes')
ax.set_xlabel(r'Frequency \\\\ $\textbf{(a)}$')
ax.set_xticks(range(0,700,100))
fig.savefig('./results/plots/free0.png')

plt.show()

