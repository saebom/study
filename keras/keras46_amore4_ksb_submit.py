import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

## 시험문제 ##
# 삼성전자와 아모레 주가로 아모레 가격 맞추기
# 가격 데이터에서 컬럼 7개 이상씩 추출(그 중 거래량은 반드시 들어가게 함)
# timesteps와 feature는 알아서 자르고
# 제공된 데이터 외 추가 데이터 사용금지
# 아모레 19일 아침 시가 맞추기 (점수배점 0.3)
# 아모레 20일 종가 맞추기 (점수배점 0.7)

#1. 데이터
path = './_data/test_amore_0718/'
amore = pd.read_csv(path +"아모레220718.csv", thousands=",", encoding='cp949')  # 마이크로소프트 'cp949' / 맥과 리눅스는 'utf8'
samsung = pd.read_csv(path +"삼성전자220718.csv", thousands=",", encoding='cp949')  # 마이크로소프트 'cp949' / 맥과 리눅스는 'utf8'

#일자 변환
amore['일자'] = pd.to_datetime(amore['일자'], infer_datetime_format=True)
samsung['일자'] = pd.to_datetime(samsung['일자'], infer_datetime_format=True)

#데이터 행 맞추기
amore = amore.drop(range(744, 3180), axis=0)
samsung = samsung.drop(range(744, 3040), axis=0)

#과거에서 현재 순으로 정렬
amore = amore.loc[::-1].reset_index(drop=True)
samsung = samsung.loc[::-1].reset_index(drop=True)

#str 타입 삭제
amore = amore.drop(columns=['전일비'], axis=1)
samsung = samsung.drop(columns=['전일비'], axis=1)

#dataframe으로 변환
amore = amore.apply(pd.to_numeric) # convert all columns of DataFrame
samsung = samsung.apply(pd.to_numeric) # convert all columns of DataFrame

#dataset 만들기
dataset_a = amore.drop(columns=['일자', 'Unnamed: 6', '금액(백만)', '신용비', '등락률', '외인(수량)', '프로그램', '외인비'], axis=1) 
dataset_a = dataset_a[['시가', '고가', '저가', '거래량', '개인', '기관', '종가']]
dataset_a = np.array(dataset_a)

dataset_s = amore.drop(columns=['일자', 'Unnamed: 6', '금액(백만)', '신용비', '등락률', '외인(수량)', '프로그램', '외인비'], axis=1) 
dataset_s = dataset_s[['시가', '고가', '저가',  '거래량', '개인', '기관', '종가']]
dataset_s = np.array(dataset_s)

time_steps = 5
y_column = 3

def split_xy(dataset, time_steps, y_column):                 
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps      # 0 + 5   = 5
        y_end_number = x_end_number + y_column    # 5 + 3 = 8
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]                   # 0 : 5 , : -1   > 0행~4행, 마지막열 뺀 전부
        tmp_y = dataset[x_end_number:y_end_number, -1]       # 5 - 1 : 7 , -1  > 마지막 열의 4~6행
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy(dataset_a, time_steps, y_column)
x2, y2 = split_xy(dataset_s, time_steps, y_column)

#train, test split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test \
= train_test_split(x1, x2, y1, y2, shuffle=False, train_size=0.9)

mean1 = np.mean(x1_train)
std1 = np.std(x1_train)
x1_train = (x1_train-mean1)/std1
x1_test = (x1_test-mean1)/std1

mean2 = np.mean(x2_train)
std2 = np.std(x2_train)
x2_train = (x2_train-mean2)/std2
x2_test = (x2_test-mean2)/std2

x1_train = x1_train.reshape(663, 5*6, 1) 
x1_test = x1_test.reshape(74, 5*6, 1)

x2_train = x2_train.reshape(663, 5*6, 1) 
x2_test = x2_test.reshape(74, 5*6, 1)

#3.#load_model
model = load_model('./_test/_0720_0346_0012-69229.2578.hdf5')

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)
print('loss :', loss) 

y_predict = model.predict([x1_test, x2_test])
print('[아모레 7월 20일 종가] 예측 : ', y_predict[-1])



# ======================================= loss 및 예측값 ===========================================
# './_test/_0720_0550_0020-49236.9414.hdf5'
# loss : [54372.21484375, 16540351488.0]
# [아모레 7월 20일 종가] 예측 :  [132510.38]
# ==================================================================================================


# ============================================ summary() ===========================================
# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            [(None, 30, 1)]      0
# __________________________________________________________________________________________________
# input_2 (InputLayer)            [(None, 30, 1)]      0
# __________________________________________________________________________________________________
# sb1 (LSTM)                      (None, 30, 100)      40800       input_1[0][0]
# __________________________________________________________________________________________________
# sb11 (LSTM)                     (None, 30, 100)      40800       input_2[0][0]
# __________________________________________________________________________________________________
# sb2 (LSTM)                      (None, 100)          80400       sb1[0][0]
# __________________________________________________________________________________________________
# sb12 (LSTM)                     (None, 100)          80400       sb11[0][0]
# __________________________________________________________________________________________________
# sb3 (Dense)                     (None, 100)          10100       sb2[0][0]
# __________________________________________________________________________________________________
# sb13 (Dense)                    (None, 100)          10100       sb12[0][0]
# __________________________________________________________________________________________________
# out_sb1 (Dense)                 (None, 1)            101         sb3[0][0]
# __________________________________________________________________________________________________
# out_sb2 (Dense)                 (None, 1)            101         sb13[0][0]
# __________________________________________________________________________________________________
# mg1 (Concatenate)               (None, 2)            0           out_sb1[0][0]
#                                                                  out_sb2[0][0]
# __________________________________________________________________________________________________
# mg2 (Dense)                     (None, 100)          300         mg1[0][0]
# __________________________________________________________________________________________________
# mg3 (Dense)                     (None, 100)          10100       mg2[0][0]
# __________________________________________________________________________________________________
# mg4 (Dense)                     (None, 100)          10100       mg3[0][0]
# __________________________________________________________________________________________________
# last1 (Dense)                   (None, 1)            101         mg4[0][0]
# ==================================================================================================
# Total params: 283,403
# Trainable params: 283,403
# Non-trainable params: 0