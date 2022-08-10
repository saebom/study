import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xg
print('xgboost 버전 : ', xg.__version__)    # xgboost 버전 :  1.6.1

'''
01. iris
02. cancer
# 03. diabets
04. wine 
05. fetch_covtype
06. digits
07. kaggle_titanic
'''

#1. 데이터
from keras.datasets import mnist   
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)  
x_test = x_test.reshape(10000, 784)   
# print(np.unique(x_train, return_counts=True))

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# LDA 
lda = LinearDiscriminantAnalysis(n_components=9)
lda.fit(x_train, y_train)
x_train = lda.transform(x_train)

print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)
print(np.unique(y_train, return_counts=True))   
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

#2. 모델
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=72)

parameters = [
    {'n_estimators':[100, 200], 'learning_rate':[0.1, 0.3, 0.001], 'max_depth':[4,5]},
    {'n_estimators':[90, 100], 'learning_rate':[0.1, 0.01], 'max_depth':[5,6], 'colsample_bytree':[0.6, 1]},
    {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.5], 'max_depth':[4,6], 
     'colsample_bytree':[0.6, 1], 'colsample_bylevel':[0.6,0.9]},
]


#2. 모델구성
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0), 
                     parameters, cv=kfold, verbose=2, 
                     refit=True, n_jobs=-1)                  

# 3. 훈련
import time
start = time.time()
model.fit(x_train, y_train) 
print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print("best_score_ : ", model.best_score_)
print("model.score : ", model.score(x_test, y_test))
end_time = time.time() 


#4. 평가, 예측
y_predict = model.predict(x_test)
print('LDA 적용 결과 : ', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적의 튠 ACC : ', accuracy_score(y_test, y_pred_best))
print('걸린시간 : ', round(end_time-start, 2), "초")



#==================================== 결과 ==================================#
# 적용 전 결과  :  0.9641
# LDA 적용 결과 :   0.9184666666666665
#============================================================================#