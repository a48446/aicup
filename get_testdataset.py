import os
import pandas as pd
from shutil import copyfile

csv_name = "test"
data_set = pd.read_csv("init_data/{}.csv".format(csv_name))
data = data_set.copy()
total = 0
for i in data["filename"]:
    if not os.path.exists('./init_data/test'):  os.makedirs('./init_data/test')
    copyfile('./init_data/train/{}'.format(i), './init_data/test/{}'.format(i))
    print("複製 train 資料夾 {} 檔案至 test 資料夾".format(i))
    total += 1
print("總計 {} 筆".format(total))

