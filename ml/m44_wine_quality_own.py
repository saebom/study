# csv로 맹그러!!!

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


#1. 데이터
path = 'D:/study_data/_data/'
data_set = pd.read_csv(path + 'winequality-white.csv', index_col=None, header=0, sep=';')

print('data_set.shape', data_set.shape)    # (4898, 12)

x = data_set.drop(['quality'], axis=1)
y = data_set['quality']

print(y.unique())
print(x.shape, y.shape)     # (4898, 11) (4898,)

# from sklearn.preprocessing import LabelEncoder
# lbe = LabelEncoder() 
# y = lbe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, shuffle=True, random_state=123)


#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from xgboost import XGBClassifier

# model = SVC()
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=1000, max_depth=100))    
# model = make_pipeline(StandardScaler(), XGBClassifier())   # 0.7020408163265306 

#3. 훈련
model.fit(x_train, y_train) # make_pipeline에서의 fit은 fit_transform이 적용됨


#4. 평가, 예측
result = model.score(x_test, y_test)   
print('model.score : ', result) 

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)



#=============================== 결 과 ==================================#
# accuracy_score :  0.7238095238095238
#========================================================================#
