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

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72
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


#==================================== pipeline 적용결과 =================================#
# model.score :  0.9649122807017544
#============================= HalvingRandomSearchCV 결과 ===============================#
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=10, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 10}
# best_score_ :  0.961111111111111
# model.score :  0.9649122807017544
# accuracy_score :  0.9649122807017544
# 최적의 튠 ACC :  0.9649122807017544
# 걸린시간 :  5.86 초
#============================= HalvingGridSearchCV 결과 ===============================#
# Fitting 5 folds for each of 22 candidates, totalling 110 fits
# 최적의 매개변수 :  RandomForestClassifier(n_jobs=4)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 4}
# best_score_ :  0.9444444444444443
# model.score :  1.0
# accuracy_score :  1.0
# 최적의 튠 ACC :  1.0
# 걸린시간 :  13.09 초
#============================== RandomizedSearchCV 결과 ===============================#
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 3}
# best_score_ :  0.9673101265822786
# model.score :  0.9707602339181286
# accuracy_score :  0.9707602339181286
# 최적의 튠 ACC :  0.9707602339181286
# 걸린시간 :  3.73 초
#================================= GridSearchCV 결과 ===================================#
# Fitting 5 folds for each of 66 candidates, totalling 330 fits
# 최적의 매개변수 :  RandomForestClassifier(n_jobs=4)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 4}
# best_score_ :  0.969873417721519
# model.score :  0.9590643274853801
# accuracy_score :  0.9590643274853801
# 최적의 튠 ACC :  0.9590643274853801
# 걸린시간 :  10.85 초
# =======================================================================================#
