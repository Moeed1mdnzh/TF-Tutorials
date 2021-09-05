import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer

def shuffle(data, num_of_feats):
    #  Shuffle the data
    np.random.shuffle(data)
    print(f"total number of features : {num_of_feats}")
    return data

def split_to_sets(data, num_of_feats):
    #  Split the data
    test_size = int(0.25 * num_of_feats) 
    train = data[test_size:]
    test = data[:test_size]
    val_size = int(0.20 * test.shape[0])
    val = test[:val_size]
    test = test[val_size:]
    for name, _set in zip(("Train", "Val", "Test"), (train, val, test)):
        print(f"{name} size : {_set.shape[0]}")
        imp_med = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_med.fit(_set)
        if name == "Train":
            train = imp_med.transform(train)
            np.save("dataset/train_distr", np.concatenate((np.mean(train, axis=0), np.std(train, axis=0))))
        if name == "Val":
            val = imp_med.transform(val)
            np.save("dataset/val_distr", np.concatenate((np.mean(val, axis=0), np.std(val, axis=0))))
        if name == "Test":
            test = imp_med.transform(test)
            np.save("dataset/test_distr", np.concatenate((np.mean(test, axis=0), np.std(test, axis=0))))
    return train, val, test

def decompose(train, val, test, column_names):
    #  Split each set into many csv files
    for name, _set in zip(("train", "val", "test"), (train, val, test)):
        for pos, i in enumerate(range(0, _set.shape[0]+129, 129)):
            sub_set = _set[i:i + 129]
            new_set = pd.DataFrame({column_names[i] : sub_set[:, i] for i in range(9)})
            file_name = "00" + str(pos)
            if pos < 100 and pos > 9:
                file_name = file_name[1:]
            elif pos > 99:
                file_name = file_name[2:]
            new_set.to_csv(f"dataset/{name}/my_{name}_{file_name}.csv", index=False)

def load(name):
    df = pd.read_csv(name) #  Load the data
    column_names = list(df.columns.values)[:9]
    df = df.to_numpy()
    data = df[:, :9]
    num_of_feats = data.shape[0]
    return data, num_of_feats, column_names
    