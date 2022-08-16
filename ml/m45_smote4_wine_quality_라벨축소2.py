import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

#1. 데이터
path = 'D:/study_data/_data/'
data_set = pd.read_csv(path + 'winequality-white.csv', index_col=None, header=0, sep=';')

print('data_set.shape', data_set.shape)    # (4898, 12)
print(data_set.describe())
print(data_set.info())

# x = data_set.drop(['quality'], axis=1)
# y = data_set['quality']

data_set2 = data_set.to_numpy()
# data_set2 = data_set.values
print(type(data_set2))
print(data_set2.shape)  # (4898, 12)

x = data_set2[:, :11]
y = data_set2[:, 11]
print(x.shape, y.shape) # (4898, 11) (4898,)
print(np.unique(y, return_counts=True))  
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
print(data_set['quality'].value_counts)
# print(y[:10])   # [6. 6. 6. 6. 6. 6. 6. 6. 6. 6.]
# print(y[1000:1010]) # [7. 5. 6. 7. 6. 6. 6. 5. 7. 6.]
print(y[:20])   # [6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 5. 5. 5. 7. 5. 7. 6. 8. 6. 5.]

for index, value in enumerate(y):
    if value == 9 : 
       y[index] = 6
    elif value == 8 : 
       y[index] = 6
    elif value == 7 : 
       y[index] = 6
    elif value == 6 : 
       y[index] = 5
    elif value == 5 : 
       y[index] = 5
    elif value == 4 : 
       y[index] = 4
    elif value == 3 : 
       y[index] = 4
    else:
        y[index] = 0

print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([ 183, 3655, 1060], dtype=int64))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, shuffle=True, train_size=0.8,
    stratify=y
)

print(pd.Series(y_train).value_counts())
# 5.0    2924
# 6.0     848
# 4.0     146
print("#========================== SMOTE 적용 후 ============================ ")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123, k_neighbors=1)  
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())
# 5.0    2924
# 4.0    2924
# 6.0    2924
#2. 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline

model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=1000, max_depth=100))    

# model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
from sklearn.metrics import accuracy_score, f1_score

y_predict = model.predict(x_test) 
score = model.score(x_test, y_test)
print('model.score : ', score)     
print('acc_score : ', accuracy_score(y_test, y_predict))
print('f1_score(macro) : ', f1_score(y_test, y_predict, average='macro'))   # f1_score는 이진분류이므로 average를 사용하여 다중분류에 사용
print('f1_score(micro) : ', f1_score(y_test, y_predict, average='micro'))   #


 
#================================== 결 과 =====================================#
# model.score :  0.7387755102040816
# acc_score :  0.7387755102040816
# f1_score(macro) :  0.6404492652222676
# f1_score(micro) :  0.7387755102040816
# ==========================================
# 인덱스 축소 후(4)
# model.score :  0.7214285714285714
# acc_score :  0.7214285714285714
# f1_score(macro) :  0.674435196840007
# f1_score(micro) :  0.7214285714285713
# ==========================================
# 인덱스 축소 후(3)
# model.score :  0.8397959183673469
# acc_score :  0.8397959183673469
# f1_score(macro) :  0.6715288240305375
# f1_score(micro) :  0.8397959183673469
#==============================================================================#
