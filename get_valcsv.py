import os
import pandas as pd

csv_name = "train"
data_set = pd.read_csv("data/{}.csv".format(csv_name))
data = data_set.copy()
times = 0
range_1 = 1
range_2 = 2
labelname = []
filename = []

for i in range(219):
    label1 = data['category'][range_1]
    label2 = data['category'][range_2]
    filename1 = data['filename'][range_1]
    filename2 = data['filename'][range_2]
    if str(filename1) == str(filename2):
        print("檔案名稱相同")
        break

    labelname.append(label1)
    labelname.append(label2)
    filename.append(filename1)
    filename.append(filename2)
    range_1 +=10
    range_2 +=10

print(labelname)
print(filename)

new_data = pd.DataFrame({   'filename':pd.Series(filename),
                            "category":pd.Series(labelname),
                            })
new_data.info()
new_data.to_csv('data/val.csv')
