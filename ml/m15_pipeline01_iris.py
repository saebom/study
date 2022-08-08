import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=1234
    )
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)   # train은 fit_transform, test는 transform으로 overfit(과적합)이 안 잡힘


#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

# model = SVC()
# model = make_pipeline(MinMaxScaler(), SVC())    # pipeline에는 scaling과 model을 정의하지 않고 사용
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())    # pipeline에는 scaling과 model을 정의하지 않고 사용


#3. 훈련
model.fit(x_train, y_train) # make_pipeline에서의 fit은 fit_transform이 적용됨


#4. 평가, 예측
result = model.score(x_test, y_test)    # make_pipeline에서의 model.score는 transform이 적용됨
print('model.score : ', result) # model.score :  1.0


#================================== pipeline 적용결과 ===================================#
# model.score :  1.0
#============================= HalvingRandomSearchCV 결과 ===============================#
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=10, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'n_estimators': 100, 'max_depth': 10}
# best_score_ :  0.9555555555555555
# model.score :  1.0
# accuracy_score :  1.0
# 최적의 튠 ACC :  1.0
# 걸린시간 :  3.21 초
#============================= HalvingGridSearchCV 결과 ===============================#
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# 최적의 매개변수 :  SVC(C=1, degree=5, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 5, 'kernel': 'linear'}
# best_score_ :  0.9777777777777779
# model.score :  1.0
# accuracy_score :  1.0
# 최적의 튠 ACC :  1.0
# 걸린시간 :  2.48 초
#============================== RandomizedSearchCV 결과 ===============================#
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=10)  
# 최적의 파라미터 :  {'n_estimators': 100, 'min_samples_leaf': 10}
# best_score_ :  0.9666666666666668
# model.score :  1.0
# accuracy_score :  1.0
# 최적의 튠 ACC :  1.0
# 걸린시간 :  3.3 초 
#================================= GridSearchCV 결과 ===================================#
# Fitting 5 folds for each of 66 candidates, totalling 330 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, n_jobs=-1)     
# 최적의 파라미터 :  {'max_depth': 6, 'n_estimators': 100, 'n_jobs': -1}
# best_score_ :  0.95
# model.score :  1.0
# accuracy_score :  1.0
# 최적의 튠 ACC :  1.0
# 걸린시간 :  10.61 초
# =======================================================================================#

