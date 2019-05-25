from data.data_proc_hk import get_data
from data.data_proc_hk import get_stop_words
from data.data_proc_hk import get_tfidfvect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import jieba
import os
import sys

data_num = 30000 # 用于训练的数据量
model_save = "./data/models/gbdt.m"
os.makedirs(os.path.dirname(model_save), exist_ok=True)


allX, labels = get_tfidfvect(data_num)
x_train, x_test, y_train, y_test = train_test_split(allX, labels, test_size=0.25)

def model_train(train_datas, train_labels):
  gbr = GradientBoostingClassifier(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)
  gbr.fit(train_datas, train_labels.ravel())
  joblib.dump(gbr, model_save)   # 保存模型
  y_gbr = gbr.predict(train_datas)
  acc_train = gbr.score(train_datas, train_labels)
  print("训练数据集上的精度是：", acc_train)

def model_test(test_data,test_labels):
  # 加载模型并预测
  gbr = joblib.load(model_save)  # 加载模型
  acc_test = gbr.score(test_data, test_labels)
  print("测试数据集上的精度是：", acc_test)

def model_pred(pred_data):
  gbr = joblib.load(model_save)
  print("result:")
  print(gbr.predict(pred_data))

model_train(x_train, y_train)
model_test(x_test, y_test)

