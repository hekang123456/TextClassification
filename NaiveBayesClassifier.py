#-*-coding:utf-8-*-

# sklearn 地址： https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes

# 最近中心分类算法
from data.data_proc_hk import get_data
from data.data_proc_hk import get_stop_words
from data.data_proc_hk import get_tfidfvect
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import svm
import jieba
import os
import sys
import pydotplus
import graphviz
import pickle


data_num = 30000 # 用于训练和测试的数据数量
model = "Gaussian" # 可选的包括了 "Gaussian", "Multinomial", "Complement", "Bernoulli"
model_save = "./data/models/"+model +"NaiveBayer.m"
os.makedirs(os.path.dirname(model_save),exist_ok=True)

allX, labels = get_tfidfvect(data_num)

x_train, x_test, y_train, y_test = train_test_split(allX, labels, test_size=0.25)

def model_train(train_datas, train_labels):
  """产生决策树。"""
  model_dict = {
    "Gaussian": GaussianNB(),
    "Multinomial":MultinomialNB(),
    "Complement":ComplementNB(),
    "Bernoulli":BernoulliNB()
  }
  global  model

  clf = model_dict[model]
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