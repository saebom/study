import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
#pip install imblearn
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('사이킷런 : ', sk.__version__)    # 1.1.2

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (178, 13) (178,) 
print(type(x))          # <class 'numpy.ndarray'>
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

print(pd.Series(y).value_counts())
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x_new = x[:-35]
y_new = y[:-35]
print(pd.Series(y_new).value_counts())
# 1    71
# 0    59
# 2    18

x_train, x_test, y_train, y_test = train_test_split(
    x_new, y_new, train_size=0.75, shuffle=True, random_state=123,
    stratify=y_new
)

print(pd.Series(y_train).value_counts())
# 1    53
# 0    44
# 2    6

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
from sklearn.metrics import accuracy_score, f1_score

y_predict = model.predict(x_test) 
score = model.score(x_test, y_test)
# print('model.score : ', score)     
print('acc_score : ', accuracy_score(y_test, y_predict))
print('f1_score(macro) : ', f1_score(y_test, y_predict, average='macro'))   # f1_score는 이진분류이므로 average를 사용하여 다중분류에 사용
# print('f1_score(micro) : ', f1_score(y_test, y_predict, average='micro'))
#==================================================================#
# 기본 결과
# acc_score :  0.9777777777777777
# f1_score(macro) :  0.9797235023041475
# ================================================
# 데이터 축소 후 (2라벨을 40개 줄인 후)
# acc_score :  0.9428571428571428
# f1_score(macro) :  0.8596176821983273
#==================================================================#

print("#========================== SMOTE 적용 후 ============================ ")
smote = SMOTE(random_state=123)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())
# 0    53
# 1    53
# 2    53

#2. 모델 #3. 훈련
model = RandomForestClassifier()
model.fit(x_train, y_train)

#4. 평가, 예측
from sklearn.metrics import accuracy_score, f1_score

y_predict = model.predict(x_test) 
score = model.score(x_test, y_test)
# print('model.score : ', score)     
print('acc_score : ', accuracy_score(y_test, y_predict))
print('f1_score(macro) : ', f1_score(y_test, y_predict, average='macro'))   # f1_score는 이진분류이므로 average를 사용하여 다중분류에 사용
# print('f1_score(micro) : ', f1_score(y_test, y_predict, average='micro'))


#================================ 결과 ================================#
# acc_score :  0.9444444444444444
# f1_score(macro) :  0.9040621266427719
# ============================ SMOTE 적용 후 ===========================
# acc_score :  0.9714285714285714
# f1_score(macro) :  0.9797235023041475
#==================================================================#
