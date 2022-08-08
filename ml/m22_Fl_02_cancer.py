import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(datasets.DESCR)
print(x.shape)  # (569, 30)

# drop_features
x = np.delete(x, [8, 9, 11, 14, 15, 16, 17, 18], axis=1)
print(x.shape)  # (569, 22)


from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72
    )
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)   # train은 fit_transform, test는 transform으로 overfit(과적합)이 안 잡힘

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
# print('model.score : ', result)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test,)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

print('==============================')
print(model, ': ', model.feature_importances_)



#결과비교

#======================================  [8, 9, 11, 14, 15, 16, 17, 18] 삭제  결과 =======================================#
# 1. DecisionTree
# 기존 acc : 0.9122807017543859
# 컬럼 삭제 후 acc : 0.9035087719298246

# 2. RandomForestClassifier
# 기존 acc : 0.9649122807017544
# 컬럼 삭제 후 acc : 0.9649122807017544
# RandomForestClassifier() :  [0.11731571 0.42726147 0.45542282]

# 3. GradientBoostingClassifier
# 기존 acc :  0.9649122807017544
# 컬럼 삭제 후 acc : 0.9649122807017544
# GradientBoostingClassifier() :  [0.02026579 0.30569661 0.6740376 ]

# 4. XGBClassifier
# 기존 acc : 0.9736842105263158
# 컬럼 삭제 후 acc :0.9736842105263158
# XGBClassifier() :  [0.02544769 0.66221166 0.31234065]
#=========================================================================================================================#
