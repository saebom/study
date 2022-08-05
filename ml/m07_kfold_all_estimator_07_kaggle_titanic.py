import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.9, shuffle=True, random_state=23
# )

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=72)

#2. 모델 구성, 훈련, 평가, 예측
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms: 
    try :
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold)
        print('ACC : ', scores, '\n cross_val_score ; ', round(np.mean(scores), 4))
    except:
        # continue
        print(name, '은 안나온 놈!!!')
        


#===================================== 결  과 ==========================================#
# ACC :  [0.82122905 0.82022472 0.79775281 0.78089888 0.75280899] 
#  cross_val_score ;  0.7946
# ACC :  [0.79888268 0.84269663 0.80898876 0.7752809  0.78651685] 
#  cross_val_score ;  0.8025
# ACC :  [0.74860335 0.78651685 0.76966292 0.80898876 0.79775281] 
#  cross_val_score ;  0.7823
# ACC :  [0.79329609 0.83146067 0.75280899 0.73595506 0.80337079] 
#  cross_val_score ;  0.7834
# ACC :  [       nan 0.83146067 0.74719101        nan 0.70786517] 
#  cross_val_score ;  nan
# ClassifierChain 은 안나온 놈!!!
# ACC :  [0.70391061 0.76404494 0.70224719 0.66853933 0.62921348] 
#  cross_val_score ;  0.6936
# ACC :  [0.77653631 0.83707865 0.74719101 0.75280899 0.79213483] 
#  cross_val_score ;  0.7811
# ACC :  [0.59776536 0.67977528 0.60674157 0.58988764 0.60674157]
#  cross_val_score ;  0.6162
# ACC :  [0.73184358 0.7752809  0.79775281 0.73595506 0.7752809 ] 
#  cross_val_score ;  0.7632
# ACC :  [0.79329609 0.83146067 0.76966292 0.7752809  0.80898876] 
#  cross_val_score ;  0.7957
# ACC :  [0.80446927 0.80337079 0.75280899 0.80337079 0.78089888] 
#  cross_val_score ;  0.789
# ACC :  [0.70949721 0.75280899 0.69662921 0.69662921 0.69101124] 
#  cross_val_score ;  0.7093
# ACC :  [0.8547486  0.84831461 0.82022472 0.83707865 0.78651685] 
#  cross_val_score ;  0.8294
# ACC :  [0.83240223 0.83707865 0.81460674 0.8258427  0.79775281] 
#  cross_val_score ;  0.8215
# ACC :  [0.65363128 0.74719101 0.6741573  0.70224719 0.69662921] 
#  cross_val_score ;  0.6948
# ACC :  [0.68156425 0.73033708 0.65168539 0.69662921 0.71910112] 
#  cross_val_score ;  0.6959
# ACC :  [0.68156425 0.73033708 0.65730337 0.69662921 0.7247191 ] 
#  cross_val_score ;  0.6981
# ACC :  [0.79888268 0.79775281 0.75842697 0.80898876 0.79213483] 
#  cross_val_score ;  0.7912
# ACC :  [0.63687151 0.78651685 0.74719101 0.79213483 0.75280899] 
#  cross_val_score ;  0.7431
# ACC :  [0.81005587 0.79775281 0.75280899 0.80898876 0.79213483] 
#  cross_val_score ;  0.7923
# ACC :  [0.81564246 0.80337079 0.75842697 0.80898876 0.79213483] 
#  cross_val_score ;  0.7957
# ACC :  [0.80446927 0.79775281 0.80337079 0.82022472 0.76966292] 
#  cross_val_score ;  0.7991
# MultiOutputClassifier 은 안나온 놈!!!
# ACC :  [0.69832402 0.75280899 0.69662921 0.67977528 0.62921348] 
#  cross_val_score ;  0.6914
# ACC :  [0.67039106 0.73033708 0.64606742 0.65730337 0.62359551] 
#  cross_val_score ;  0.6655
# ACC :  [0.80446927 0.78651685 0.76404494 0.83146067 0.80898876] 
#  cross_val_score ;  0.7991
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# ACC :  [0.75418994 0.81460674 0.74157303 0.7247191  0.71348315] 
#  cross_val_score ;  0.7497
# ACC :  [0.75977654 0.41011236 0.6011236  0.70786517 0.67977528] 
#  cross_val_score ;  0.6317
# ACC :  [0.83798883 0.78651685 0.75280899 0.81460674 0.81460674] 
#  cross_val_score ;  0.8013
# ACC :  [nan nan nan nan nan] 
#  cross_val_score ;  nan
# ACC :  [0.7877095  0.85955056 0.78089888 0.79213483 0.79775281] 
#  cross_val_score ;  0.8036
# ACC :  [0.79888268 0.79775281 0.76404494 0.82022472 0.79775281] 
#  cross_val_score ;  0.7957
# ACC :  [0.79888268 0.79775281 0.76404494 0.82022472 0.79775281] 
#  cross_val_score ;  0.7957
# ACC :  [0.82681564 0.74157303 0.75842697 0.78651685 0.79775281] 
#  cross_val_score ;  0.7822
# ACC :  [0.67597765 0.74157303 0.66292135 0.66853933 0.63483146] 
#  cross_val_score ;  0.6768
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
#=======================================================================================#

