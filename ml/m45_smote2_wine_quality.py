# 실습!! 시작!!!
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


#1. 데이터
path = 'D:/study_data/_data/'
data_set = pd.read_csv(path + 'winequality-white.csv', index_col=None, header=0, sep=';')

print('data_set.shape', data_set.shape)    # (4898, 12)
print(data_set.describe())
print(data_set.info())

data_set2 = data_set.to_numpy()
print(type(data_set2))
print(data_set2.shape)  # (4898, 12)

x = data_set2[:, :11]
y = data_set2[:, 11]
print(x.shape, y.shape) # (4898, 11) (4898,)
print(np.unique(y, return_counts=True))  

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, shuffle=True, train_size=0.8,
    stratify=y
)
print(pd.Series(y_train).value_counts())
# 6.0    1758
# 5.0    1166
# 7.0     704
# 8.0     140
# 4.0     130
# 3.0      16
# 9.0       4

print("#========================== SMOTE 적용 후 ============================ ")
smote = SMOTE(random_state=123, k_neighbors=3) 
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())
# 4.0    1758
# 5.0    1758
# 6.0    1758
# 7.0    1758
# 8.0    1758
# 3.0    1758
# 9.0    1758

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


#=========================== 결과 ===============================#
# k_neighbors=3
# # acc_score :  0.6806122448979591
# f1_score(macro) :  0.4331402529636481
# ==========================================
# k_neighbors=2
# acc_score :  0.6918367346938775
# f1_score(macro) :  0.43481102206372696
#================================================================#