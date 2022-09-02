import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time


#1. 데이터
path = './_data/bokchoy/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))
test_input_list = sorted(glob.glob(path + 'test_input/*.csv'))
test_target_list = sorted(glob.glob(path + 'test_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8
print(len(test_input_list))
print(len(test_target_list))


def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)
    print('끗.')

train_data, label_data = aaa(train_input_list, train_target_list) 

print(train_data[0])
print(len(train_data), len(label_data)) # 1607 1607
print(len(train_data[0]))   # 1440
print(label_data)   # 1440

x_train = train_data
y_train = label_data
print(x_train.shape, y_train.shape)   # (1607, 1440, 37) (1607,)

train_data, label_data = aaa(val_input_list, val_target_list)
x_val = train_data
y_val = label_data
print(x_val.shape, y_val.shape)   # (195, 1440, 37) (195,)

train_data, label_data = aaa(test_input_list, test_target_list)
x_test = train_data
y_test = label_data
print(x_test.shape, y_test.shape)   # (206, 1440, 37) (206,)

path = 'C:/study/_data/bokchoy/_save/'
datalist = [train_data, label_data, x_train, y_train, x_val, y_val, x_test, y_test]
import joblib
joblib.dump(datalist, path+'m46_save.dat')



