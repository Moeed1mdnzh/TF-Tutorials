import os
import numpy as np 
import pandas as pd 


df = pd.read_csv("housing.csv") #  Load the data
column_names = list(df.columns.values)[:9]
df = df.to_numpy()
data = df[:, :9]
 
np.random.shuffle(data) #  Shuffle the data
num_of_feats = data.shape[0]
print(f"total number of features : {num_of_feats}")

#  Split the data
test_size = int(0.25 * num_of_feats) 
train = data[test_size:]
test = data[:test_size]
val_size = int(0.20 * test.shape[0])
val = test[:val_size]
test = test[val_size:]
for name, size in zip(("Train", "Val", "Test"), (train, val, test)):
	print(f"{name} size : {size.shape[0]}")

#  Split each set into many csv files
os.system(r"mkdir dataset\train")
os.system(r"mkdir dataset\val")
os.system(r"mkdir dataset\test")
for name, _set in zip(("train", "val", "test"), (train, val, test)):
	for pos, i in enumerate(range(0, _set.shape[0]+129, 129)):
		sub_set = _set[i:i + 129]
		new_set = pd.DataFrame({column_names[i] : sub_set[:, i] for i in range(9)})
		file_name = "0" + str(pos)
		new_set.to_csv(f"dataset/{name}/my_{name}_{file_name if pos < 10 else pos}.csv", index=False)