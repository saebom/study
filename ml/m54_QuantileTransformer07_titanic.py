
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_boston, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time


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

# 라벨인코딩
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=7777, stratify=y
    )


# 스케일링
sts = StandardScaler() 
mms = MinMaxScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()
qtf = QuantileTransformer() 
ptf1 = PowerTransformer(method='yeo-johnson')
ptf2 = PowerTransformer(method='box-cox')

scalers = [sts, mms, mas, rbs, qtf, ptf1, ptf2]
for scaler in scalers:
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = LGBMClassifier()
    # model = XGBClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    result = r2_score(y_test, y_predict)
    scale_name = scaler.__class__.__name__
    print('{0} 결과 : {1:.4f}'.format(scale_name, result), )
    

#=================================== 결과 =====================================#
# StandardScaler 결과 : 0.3397
# MinMaxScaler 결과 : 0.3397
# MaxAbsScaler 결과 : 0.3397
# RobustScaler 결과 : 0.3161
# QuantileTransformer 결과 : 0.3397
# PowerTransformer 결과 : 0.3397
# ValueError: The Box-Cox transformation can only be applied to strictly positive data
#==============================================================================#