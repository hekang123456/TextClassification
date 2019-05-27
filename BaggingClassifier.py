#-*-coding:utf-8-*-

# sklearn 地址： https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator
# bagging 方法
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from data.data_proc_hk import get_data
from data.data_proc_hk import get_stop_words
from data.data_proc_hk import get_tfidfvect
import numpy as np
import pandas as pd
import jieba
import os
import sys
import pydotplus
import graphviz
import pickle


data_num = 3000 # 用于训练和测试的数据数量
sub_classifier = "GaussianNB" # 可选一些子分类模型， 包括 naive bayes； SVM, tree， 最近邻 等
model_save = "./data/models/BaggingBayer.m"
os.makedirs(os.path.dirname(model_save),exist_ok=True)

allX, labels = get_tfidfvect(data_num)

x_train, x_test, y_train, y_test = train_test_split(allX, labels, test_size=0.25)

def model_train(train_datas, train_labels):
  """产生决策树。"""
  clf = BaggingClassifier(
    GaussianNB(),  # 基本的分类模型。
    max_samples= 0.5, # 每次训练基本分类器从整体的训练数据中采样的比例。
    max_features = 0.5, # 每次从样本的特征中选择多少的特征进行训练。 可以是百分比也可以是整数。
  )

  train_datas = train_datas.toarray()
  model = clf.fit(train_datas, train_labels)

  # 保存产生的模型
  with open(model_save, 'wb') as f:
    pickle.dump(clf, f)

  train_acc = clf.score(train_datas, train_labels)
  print("训练集上的精度是：", train_acc)

def model_test(test_data,test_labels):
  # 加载模型并预测
  with open(model_save, 'rb') as f:
    dtree = pickle.load(f)  # 加载模型
  test_data = test_data.toarray()
  acc_test = dtree.score(test_data, test_labels)
  print("测试数据集上的精度是：", acc_test)

def model_pred(pred_data):
  with open(model_save, 'rb') as f:
    dtree = pickle.load(f)  # 加载模型

  pred_data = pred_data.toarray()
  print(dtree.predict(pred_data))

model_train(x_train, y_train)
model_test(x_test, y_test)
model_pred(x_test)