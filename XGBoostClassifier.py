#-*-coding:utf-8-*-
from xgboost import XGBClassifier
from data.data_proc_hk import get_data
from data.data_proc_hk import get_stop_words
from data.data_proc_hk import get_tfidfvect
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import jieba
import os
import sys
import pydotplus
import graphviz
import pickle

from pandas import DataFrame

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

data_num = 100000 # 用于训练和测试的数据数量
model_save = "./data/models/XGBoost.m"
os.makedirs(os.path.dirname(model_save),exist_ok=True)

allX, labels = get_tfidfvect(data_num)

labSet = set(labels)
lab_dict = {}
for i, item in enumerate(labSet):
  lab_dict[item] = i
labels = [lab_dict[item] for item in labels]

x_train, x_test, y_train, y_test = train_test_split(allX, labels, test_size=0.25)

def model_train(train_datas, train_labels):
  """产生决策树。"""

  clf = XGBClassifier(
    n_estimators=30,  # 三十棵树
    learning_rate=0.3,
    max_depth=3,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
  )

  model = clf.fit(train_datas, train_labels)

  # 保存产生的模型
  with open(model_save, 'wb') as f:
    pickle.dump(clf, f)

  train_acc = clf.score(train_datas, train_labels)
  print("训练集上的精度是：", train_acc)

def model_test(test_data,test_labels):
  # 加载模型并预测
  with open(model_save, 'rb') as f:
    xgb = pickle.load(f)  # 加载模型
  acc_test = xgb.score(test_data, test_labels)
  print("测试数据集上的精度是：", acc_test)

def model_pred(pred_data):
  with open(model_save, 'rb') as f:
    xgb = pickle.load(f)  # 加载模型
  print("预测的结果是：")
  print(xgb.predict(pred_data))

model_train(x_train, y_train)
model_test(x_test, y_test)
model_pred(x_test)


