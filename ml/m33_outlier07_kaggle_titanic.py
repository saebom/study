import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. 데이터

path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

submission = pd.read_csv(path + 'gender_submission.csv')
# print('train.shape : ', train_set.shape, 'test.shape : ', test_set.shape)   
# train.shape :  (891, 11) test.shape :  (418, 10)

# 데이터 전처리
train_set[['Pclass', 'Survived']].groupby(['Pclass'], 
                                          as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Sex', 'Survived']].groupby(['Sex'], 
                                       as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['SibSp', 'Survived']].groupby(['SibSp'], 
                                         as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Parch', 'Survived']].groupby(['Parch'], 
                                         as_index=False).mean().sort_values(by='Survived', ascending=False)

# 결측치 확인
# print(train_set.isnull().sum())  #Age 177, Cabin 687, Embarked 2
# print(test_set.isnull().sum())   #Age 86, Fare 1, Cabin 327

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
# print(train_set.isnull().sum())  #Age 177, Embarked 2
# print(test_set.isnull().sum())   #Age 86, Fare 1

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
    print("med",np.median(out))
    return np.median(out)
    
col = "Fare"
medOutlier = outliers(train_set,col)
train_set[train_set[col] >= medOutlier]
# print(train_set.describe()["Fare"])

col = "Age"
medOutlier = outliers(train_set,col)
train_set[train_set[col] >= medOutlier]
# print(train_set.describe()["Age"])

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

#2. 모델구성
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# model = RandomForestClassifier()
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)


#3. 훈련
model.fit(x_train, y_train) # make_pipeline에서의 fit은 fit_transform이 적용됨


#4. 평가, 예측
result = model.score(x_test, y_test)    # make_pipeline에서의 model.score는 transform이 적용됨
# print('model.score : ', result) 

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test,)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)


#==================================== 결과 ==================================#
# 기존 acc :  0.8268156424581006
# 결측치 및 이상치 처리 후 acc : 0.8547486033519553
#============================================================================#