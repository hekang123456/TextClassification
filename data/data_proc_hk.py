#-*-coding:utf-8-*-
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import jieba
## 填写一些配置信息

data_path = os.path.join(os.path.dirname(__file__), "toutiao_cat_data.txt")

def get_data(my_path = None):
  if not my_path: my_path = data_path
  result = {}
  with open(my_path, 'r', encoding='utf-8') as f:
    for line in f:
      tmps = line.split("_!_")
      if len(tmps) == 5:
        result[tmps[0]] = tmps[1:]
  return result

def get_stop_words(my_path = None):
  result = []
  if my_path is None:
    my_path = os.path.join(os.path.dirname(__file__), "chinese_stop_words.hk")
  with open(my_path, 'r', encoding='utf-8') as f:
    for line in f:
      result.append(line.strip())
  return result

def get_tfidfvect(data_num= 10000) :
  """返回 tf-idf 的矩阵以及，对应的标签

  data_num: 选择训练文本的数量，最多是 382688 条训练数据
  """

  stop_words = get_stop_words()

  def data_proc():
    all_data = get_data()
    tmp = []
    for key in all_data:
      val = all_data[key]
      tmp.append(val)
    all_data = tmp

    texts = [item[2] + item[3] for item in all_data]
    labels = [item[1] for item in all_data]

    return np.array(texts), np.array(labels)

  texts, labels = data_proc()
  texts = texts[:data_num]
  labels = labels[:data_num]

  def getTfIdf(all_texts):
    """获取文本的 tf-idf 表示， 每个句子用一个很大的向量表示，向量中的值表示对应这个位置的 tf-idf 值"""
    # 一行表示一个文本， 分词之间按照空格隔开。
    result = []
    for line in all_texts:
      tmp = list(jieba.cut(line.strip()))
      tmp = [it for it in tmp if it not in stop_words]
      tmp = " ".join(tmp)
      result.append(tmp)

    # 统计词频矩阵, 得到的内容 ： (句子id,  单词在词表中的id)  单词出现的次数
    freWord = CountVectorizer()
    tf = freWord.fit_transform(result)
    # 统计每个词的 tf-idf 值， 得到的内容： (句子id, 单词在词表中的id) 对应这个单词的 tf-idf 值
    tfidftrans = TfidfTransformer()
    tfidf = tfidftrans.fit_transform(tf)

    return tfidf

  allX = getTfIdf(texts) # 可以通过 toarray 方法转为array.
  return allX, labels

if __name__ == "__main__":
  allx , labels = get_tfidfvect()