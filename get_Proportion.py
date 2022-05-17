import os
import pandas as pd

csv_name = "val"
data_set = pd.read_csv("data/{}.csv".format(csv_name))
data = data_set.copy()

val_total = len(data["category"])
print("總計 {} 筆".format(val_total))

csv_name = "train"
data_set = pd.read_csv("data/{}.csv".format(csv_name))
data = data_set.copy()
train_total = len(data["category"])
print("總計 {} 筆".format(train_total))
print("資料集比例是 {} %".format(val_total/train_total*100))