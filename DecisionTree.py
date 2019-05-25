#-*-coding:utf-8-*-
from data.data_proc_hk import get_data
from data.data_proc_hk import get_stop_words
from data.data_proc_hk import get_tfidfvect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.externals import joblib
import jieba
import os
import sys
import pydotplus
import graphviz
import pickle

data_num = 30000 # 用于训练的数据量
model_save = "./data/models/DecisionTree.m"
vision_path = "./data/vision/decision_tree.pdf"
os.makedirs(os.path.dirname(model_save),exist_ok=True)
os.makedirs(os.path.dirname(vision_path),exist_ok=True)


allX, labels = get_tfidfvect(data_num)
x_train, x_test, y_train, y_test = train_test_split(allX, labels, test_size=0.25)

def model_train(train_datas, train_labels):
  """产生决策树。"""
  clf = tree.DecisionTreeClassifier(
    criterion="gini", #  采用 基尼指数 或者 entropy
    splitter="best", # 或者“random” 在每个结点选择最好的分类点或者随机的
    max_depth=None,  # 树的最大深度
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.,
    min_impurity_split=None,
    class_weight=None,
    presort=False
  )
  model = clf.fit(train_datas, train_labels)


  # 保存产生的数
  with open(model_save, 'wb') as f:
    pickle.dump(clf, f)

  # 生成可视化文件
  dot_data = tree.export_graphviz(model, None, max_depth=100)
  graph = pydotplus.graph_from_dot_data(dot_data)
  graph.write_pdf(vision_path)
  train_acc = clf.score(train_datas, train_labels)
  print("训练集上的精度是：", train_acc)

def model_test(test_data,test_labels):
  # 加载模型并预测
  with open(model_save, 'rb') as f:
    dtree = pickle.load(f)  # 加载模型
  acc_test = dtree.score(test_data, test_labels)
  print("测试数据集上的精度是：", acc_test)

def model_pred(pred_data):
  with open(model_save, 'rb') as f:
    dtree = pickle.load(f)  # 加载模型
  print(dtree.predict(pred_data))

model_train(x_train, y_train)
model_test(x_test, y_test)
model_pred(x_test)

