# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 07:17
# @Author  : FAN FAN
# @Site    : 
# @File    : draw_williams_plot.py
# @Software: PyCharm
from plottools.williams_plot import *
import pandas as pd
import re

path = 'library/Ensembles/HCOM_FraGAT_0/'
name = 'Ensemble_0_HCOM_FraGAT_compound_descriptors_1629_OP98_2022-05-17-05-39'
df = pd.read_csv(path + name + '.csv', sep=',', encoding='windows=1250')
n_descriptors = int(df.columns[-1]) + 1

df_train = df[df['Tag'] == 'Train']
df_val = df[df['Tag'] == 'Val']
df_test = df[df['Tag'] == 'Test']

op_id = int(re.findall(r'OP(.+)_', name)[0])

#op_train_des = df_train.iloc[:, -64:]
op_train_des = df_train
df_train['Residual'] = df_train['Predict_' + str(op_id)] - df_train['Target']
df_train['Predict'] = df_train['Predict_' + str(op_id)]
for id in range(100):
    df_train.pop('Predict_' + str(id))
#op_val_des = df_val.iloc[:, -64:]
op_val_des = df_val
df_val['Residual'] = df_val['Predict_' + str(op_id)] - df_val['Target']
df_val['Predict'] = df_val['Predict_' + str(op_id)]
for id in range(100):
    df_val.pop('Predict_' + str(id))
#op_test_des = df_test.iloc[:, -64:]
op_test_des = df_test
df_test['Residual'] = df_test['Predict_' + str(op_id)] - df_test['Target']
df_test['Predict'] = df_test['Predict_' + str(op_id)]
for id in range(100):
    df_test.pop('Predict_' + str(id))

draw(df_train, df_val, df_test, n_descriptors, path)