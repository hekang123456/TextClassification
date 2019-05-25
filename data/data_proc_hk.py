#-*-coding:utf-8-*-

## 填写一些配置信息

data_path = r"toutiao_cat_data.txt"


def get_data(my_path = None):
  if not my_path: my_path = data_path
  result = {}
  with open(my_path, 'r', encoding='utf-8') as f:
    for line in f:
      tmps = line.split("_!_")
      if len(tmps) == 5:
        result[tmps[0]] = tmps[1:]
  return result
