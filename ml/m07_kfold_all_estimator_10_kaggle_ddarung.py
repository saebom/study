import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv',   # 예측에서 사용!!
                       index_col=0)

#### 결측치 처리 ####
test_set = test_set.fillna(method='ffill')
train_set = train_set.dropna()  # nan 값 삭제

x = train_set.drop(['count'], axis=1)
y = train_set['count']

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=72
# )

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)

#2. 모델 구성, 훈련, 평가, 예측
# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms: 
    try :
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold)
        print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))
    except:
        # continue
        print(name, '은 안나온 놈!!!'
        
        


#===================================== 결  과 ==========================================#


#=======================================================================================#

