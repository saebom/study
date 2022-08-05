import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, cross_val_score


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

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)

#2. 모델구성
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

#3.4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))



#===================================== 결  과 ==========================================#
# ACC :  [0.79905433 0.66945299 0.77801785 0.80136858 0.76786542] 
#  cross_val_score :  0.7632
#=======================================================================================#

