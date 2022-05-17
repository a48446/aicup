import os
import pandas as pd

csv_name = "train"
data_set = pd.read_csv("init_data/{}.csv".format(csv_name))
data = data_set.copy()
times = 0
range_1 = 1
labelname = []
filename = []

for i in range(219):
    label1 = data['category'][range_1]
    filename1 = data['filename'][range_1]
    labelname.append(label1)
    filename.append(filename1)
    range_1 +=10

print(labelname)
print(filename)

new_data = pd.DataFrame({   'filename':pd.Series(filename),
                            "category":pd.Series(labelname),
                            })
new_data.info()
new_data.to_csv('init_data/test.csv')