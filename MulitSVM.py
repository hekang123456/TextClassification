#-*-coding:utf-8-*-
# sklearn 地址： https://scikit-learn.org/stable/modules/svm.html#multi-class-classification
from data.data_proc_hk import get_data
from data.data_proc_hk import get_stop_words
from data.data_proc_hk import get_tfidfvect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import jieba
import os
import sys
import pydotplus
import graphviz
import pickle

data_num = 3000 # 用于训练的数据量
model_save = "./data/models/MulitSVM.m"
os.makedirs(os.path.dirname(model_save),exist_ok=True)

allX, labels = get_tfidfvect(data_num)

x_train, x_test, y_train, y_test = train_test_split(allX, labels, test_size=0.25)


def model_train(train_datas, train_labels):
  """产生决策树。"""
  clf = svm.SVC(
    gamma='scale', decision_function_shape='ovo'
  )

  trains = []
  for item in train_datas.toarray():
    trains.append(item.tolist())

  train_datas = trains
  model = clf.fit(train_datas, train_labels)

  # 决策面的个数与决策函数的选择是有关的， 当 选择 'ovo' 的时候个数是 n(n-1)/2
  dec = clf.decision_function([[1]])
  print("决策面的个数是：", dec.shape[1])

  # 保存产生的数
  with open(model_save, 'wb') as f:
    pickle.dump(clf, f)

  # 生成可视化文件
  train_acc = clf.score(train_datas, train_labels)
  print("训练集上的精度是：", train_acc)

def model_test(test_data,test_labels):
  tests= []
  for item in test_data.toarray():
    tests.append(item.tolist())
  test_data = tests

  # 加载模型并预测
  with open(model_save, 'rb') as f:
    dtree = pickle.load(f)  # 加载模型
  acc_test = dtree.score(test_data, test_labels)
  print("测试数据集上的精度是：", acc_test)

def model_pred(pred_data):
  tests= []
  for item in pred_data.toarray():
    tests.append(item.tolist())
  pred_data = tests

  with open(model_save, 'rb') as f:
    dtree = pickle.load(f)  # 加载模型
  print(dtree.predict(pred_data))

model_train(x_train, y_train)
model_test(x_test, y_test)
model_pred(x_test)