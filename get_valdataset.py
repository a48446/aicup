import os
import pandas as pd
from shutil import copyfile

csv_name = "val"
data_set = pd.read_csv("data/{}.csv".format(csv_name))
data = data_set.copy()
total = 0
for i in data["filename"]:
    copyfile('./data//train/{}'.format(i), './data/val/{}'.format(i))
    print("複製 train 資料夾 {} 檔案至 val 資料夾".format(i))
    total += 1
print("總計 {} 筆".format(total))

