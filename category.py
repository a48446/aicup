import os
import pandas as pd
from shutil import copyfile


mode_name = "train"
mode_name2 = "val"
mode_name3 = "test"

data_set = pd.read_csv("init_data/{}.csv".format(mode_name))
valdata_set = pd.read_csv("init_data/{}.csv".format(mode_name2))
testdata_set = pd.read_csv("init_data/{}.csv".format(mode_name3))
data = data_set.copy()
val_data = valdata_set.copy()
test_data = testdata_set.copy()
range_1 = 0
category_number = []
train_total = 0
val_total = 0
test_total = 0

for i in range(219):
    label1 = data['category'][range_1]
    category_number.append(label1)
    range_1 += 10

for i,v in  enumerate(data['category']):
    if v == category_number[v]:
        if not os.path.exists('./data/{}/category'.format(mode_name)+ str(v)): 
            os.makedirs('./data/{}/category'.format(mode_name)+ str(v))
        copyfile('./init_data/{}/{}'.format(mode_name,data['filename'][i]), './data/{}/category{}/{}'.format(mode_name,v,data['filename'][i]))
        train_total +=1
        print("建立{}集，分類第{}種花卉至對應category{}資料夾，總計有{}張".format(mode_name,v,v,train_total))

for i,v in  enumerate(val_data['category']):
    if v == category_number[v]:
        if not os.path.exists('./data/{}/category'.format(mode_name2)+ str(v)): 
            os.makedirs('./data/{}/category'.format(mode_name2)+ str(v))
        copyfile('./init_data/{}/{}'.format(mode_name2,val_data['filename'][i]), './data/{}/category{}/{}'.format(mode_name2,v,val_data['filename'][i]))
        val_total +=1
        print("建立{}集，分類第{}種花卉至對應category{}資料夾，總計有{}張".format(mode_name2,v,v,val_total))

for i,v in  enumerate(test_data['category']):
    if v == category_number[v]:
        if not os.path.exists('./data/{}/category'.format(mode_name3)+ str(v)): 
            os.makedirs('./data/{}/category'.format(mode_name3)+ str(v))
        copyfile('./init_data/{}/{}'.format(mode_name3,test_data['filename'][i]), './data/{}/category{}/{}'.format(mode_name3,v,test_data['filename'][i]))
        test_total +=1
        print("建立{}集，分類第{}種花卉至對應category{}資料夾，總計有{}張".format(mode_name3,v,v,test_total))