import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. 데이터

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

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = {'n_estimators': [100],
              'learning_rate' : [0.1, 0.2],
              'max_depth' : [3,4,5], #default 6 => 통상 max는 4정도에서 성능이 좋다
              'gamma': [1,2],
              'min_child_weight': [1,5],
              'subsample' : [0.7,1],
              'colsample_bytree' : [0.7,1],
              'colsample_bylevel' : [0.7,1],
              'colsample_bynode' : [0.7,1],
              'reg_alpha' : [0, 0.1],
              'reg_lambda' : [0, 0.1],
              }  

#2. 모델
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV

# xgb = XGBClassifier(random_state=123)
xgb = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)    
print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)
print('acc : ', result)


#=================================== 결과 =====================================#
# 최상의 매개변수 :  {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 
# 'gamma': 1, 'learning_rate': 0.2, 'max_depth': 4, 'min_child_weight': 5, 
# 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 0, 'subsample': 0.7}
# 최상의 점수 :  0.44556360303288134
# acc :  0.5011636639140128
#==============================================================================#
