import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터

path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

submission = pd.read_csv(path + 'gender_submission.csv')
print('train.shape, test.shape, submit.shape', 
      train_set.shape, test_set.shape, submission.shape)    # (891, 11) (418, 10) (418, 2)

# 데이터 전처리
train_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Ticket, Cabin, Name 삭제
train_set = train_set.drop(['Ticket', 'Cabin'], axis=1)
test_set = test_set.drop(['Ticket', 'Cabin'], axis=1)

train_set = train_set.drop(['Name'], axis=1)
test_set = test_set.drop(['Name'], axis=1)

# Age NaN값 변환
train_set['Age'] = train_set['Age'].fillna(train_set.Age.dropna().mode()[0])
test_set['Age'] = test_set['Age'].fillna(train_set.Age.dropna().mode()[0])


# Embarked, Sex NaN값 및 Object => int 변환
train_set['Embarked'] = train_set['Embarked'].fillna(train_set.Embarked.dropna().mode()[0])
train_set['Embarked'] = train_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

test_set['Embarked'] = test_set['Embarked'].fillna(test_set.Embarked.dropna().mode()[0])
test_set['Embarked'] = test_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

train_set['Sex'] = train_set['Sex'].fillna(train_set.Sex.dropna().mode()[0])
train_set['Sex'] = train_set['Sex'].map({'male':0, 'female':1}).astype(int)

test_set['Sex'] = test_set['Sex'].fillna(test_set.Sex.dropna().mode()[0])
test_set['Sex'] = test_set['Sex'].map({'male':0, 'female':1}).astype(int)

print(train_set.shape, test_set.shape)  # (891, 8) (418, 7)
print(train_set.head(5))
print(test_set.head(5))
print(train_set.isnull().sum())  
print(test_set.isnull().sum())  

# x, y 데이터
x = train_set.drop(['Survived'], axis=1)
print(x)
print(x.columns)    # 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'

y = train_set['Survived']
print(y)
print(y.shape)  # (891,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=23
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

'''모델 : RandomForestClassifier의 파라미터 ========================================================

parameters = [
    {'n_estimators' : [100, 200]},  #n_estimators 는 epoch
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]}, 
    {'n_jobs' : [-1, 2, 4]}
]

# 파라미터 조합으로 2개 이상 엮을 것
# ================================================================================================'''
parameters = [
    {'n_estimators' : [100, 200], 'max_depth':[6, 8, 10, 12], 'min_samples_split':[2, 3, 5, 10], 
     'n_jobs' : [-1, 2, 4]},  #n_estimators 는 epoch
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100, 200], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]}, 
    {'n_estimators' : [100, 200],'n_jobs' : [-1, 2, 4]}
]

#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression     #LogisticRegression은 분류모델
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier(n_estimators=100203, max_depth=6, n_jobs=2) 
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, 
#                      refit=True, n_jobs=-1)                  
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, 
                     refit=True, n_jobs=-1)   


# 4. 평가, 예측
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
print('accuracy_score : ', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적의 튠 ACC : ', accuracy_score(y_test, y_pred_best))
print('걸린시간 : ', round(end_time-start, 2), "초")

        


#============================== RandomizedSearchCV 결과 ===============================#
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=10, min_samples_split=5, n_estimators=200,
#                        n_jobs=4)
# 최적의 파라미터 :  {'n_jobs': 4, 'n_estimators': 200, 'min_samples_split': 5, 'max_depth': 10}
# best_score_ :  0.8264751552795031
# model.score :  0.8444444444444444
# accuracy_score :  0.8444444444444444
# 최적의 튠 ACC :  0.8444444444444444
# 걸린시간 :  4.7 초
#================================= GridSearchCV 결과 ===================================#
# Fitting 5 folds for each of 138 candidates, totalling 690 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=10, min_samples_split=3)
# 최적의 파라미터 :  {'max_depth': 10, 'min_samples_split': 3}
# best_score_ :  0.8327096273291927
# model.score :  0.8444444444444444
# accuracy_score :  0.8444444444444444
# 최적의 튠 ACC :  0.8444444444444444
# 걸린시간 :  26.63 초
# =======================================================================================#


