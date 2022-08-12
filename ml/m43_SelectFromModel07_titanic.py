from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd


#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'gender_submission.csv')

# 데이터 전처리
train_set[['Pclass', 'Survived']].groupby(['Pclass'], 
                                          as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Sex', 'Survived']].groupby(['Sex'], 
                                       as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['SibSp', 'Survived']].groupby(['SibSp'], 
                                         as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Parch', 'Survived']].groupby(['Parch'], 
                                         as_index=False).mean().sort_values(by='Survived', ascending=False)


# Ticket, Cabin, Name 삭제
train_set = train_set.drop(['Ticket', 'Cabin'], axis=1)
test_set = test_set.drop(['Ticket', 'Cabin'], axis=1)

train_set = train_set.drop(['Name'], axis=1)
test_set = test_set.drop(['Name'], axis=1)
# print('train.shape : ', train_set.shape, 'test.shape : ', test_set.shape) #(891, 8) (418, 7)

# Embarked, Sex Object => float 변환
train_set['Embarked'] = train_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(float)
test_set['Embarked'] = test_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(float)

train_set['Sex'] = train_set['Sex'].map({'male':0, 'female':1}).astype(float)
test_set['Sex'] = test_set['Sex'].map({'male':0, 'female':1}).astype(float)

train_set = train_set.drop(['Parch'], axis=1)
print(train_set.columns)    # Index(['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked'], dtype='object')

# outliers 처리
def outliers(df, col):
    out = []
    m = np.mean(df[col])
    sd = np.std(df[col])
    
    for i in df[col]: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
            
    print("Outliers:",out)
    print("med",np.min(out))
    return np.min(out)
    
col = "Fare"
medOutlier = outliers(train_set,col)
train_set[train_set[col] >= medOutlier]

# x, y 데이터
x = train_set.drop(['Survived'], axis=1)
print(x.shape)  # (891, 6)
print(x.columns)    # 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'

y = train_set['Survived']
print(y)
print(y.shape)  # (891,)

# IterativeImputer() 결측치 처리
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state=72)
imputer.fit(x)
x = imputer.transform(x)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators': [100],
              'learning_rate' : [0.1],
              'max_depth' : [3], 
              'gamma': [1],
              'min_child_weight': [1],
              'subsample' : [1],
              'colsample_bytree' : [1],
              'colsample_bylevel' : [1],
              'colsample_bynode' : [1],
              'reg_alpha' : [0],
              'reg_lambda' : [1]
              }  

#2. 모델
model = XGBClassifier(random_state=123, 
                      n_estimators=1000, 
                      learning_rate = 0.1,
                      max_depth = 6, 
                      gamma= 1,
                    )

#3. 훈련
model.fit(x_train, y_train, 
          early_stopping_rounds=100,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='error',
          )


#4. 평가, 예측
result = model.score(x_test, y_test)    
print('acc : ', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", acc)

print(model.feature_importances_)

thresholds = model.feature_importances_
print("=====================================")
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)   
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBClassifier(#n_jobs=-1, 
                                   random_state=123, 
                                   n_estimators=1000, 
                                   learning_rate = 0.1,
                                   max_depth = 6, 
                                   gamma= 1,)
    
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, Acc:%.2f%%"
          %(thresh, select_x_train.shape[1], score*100))


#=================================== 결과 =====================================#
# acc :  0.8547486033519553
# 진짜 최종 test 점수 :  0.8547486033519553
# [0.14403756 0.66788495 0.04083901 0.06650154 0.04772155 0.0330153 ]
# =====================================
# (712, 2) (179, 2)
# Thresh=0.144, n=2, Acc:76.54%
# (712, 1) (179, 1)
# Thresh=0.668, n=1, Acc:76.54%
# (712, 5) (179, 5)
# Thresh=0.041, n=5, Acc:84.36%
# (712, 3) (179, 3)
# Thresh=0.067, n=3, Acc:78.77%
# (712, 4) (179, 4)
# Thresh=0.048, n=4, Acc:79.89%
# (712, 6) (179, 6)
# Thresh=0.033, n=6, Acc:85.47%
#==============================================================================#