import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
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
print(x.shape)  # (891, 7)
print(x.columns)    # 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'

y = train_set['Survived']
print(y)
print(y.shape)  # (891,)

# drop_features
# x = np.delete(x, [0, 2, 7, 8, 11, 14, 15, 16, 17, 22, 23, 25, 
#                   31, 32, 33, 35, 38, 39, 40, 46, 47, 48, 55, 56, 57, 63], axis=1)
# print(x.shape)  # (1797, 38)


from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72
    )
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)   # train은 fit_transform, test는 transform으로 overfit(과적합)이 안 잡힘



#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()



#3. 훈련
model.fit(x_train, y_train) # make_pipeline에서의 fit은 fit_transform이 적용됨


#4. 평가, 예측
result = model.score(x_test, y_test)    # make_pipeline에서의 model.score는 transform이 적용됨
# print('model.score : ', result) 

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test,)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

print('==============================')
print(model, ': ', model.feature_importances_)


#결과비교

#======================================  [] 삭제  결과 =======================================#
# 1. DecisionTree
# 기존 acc : 0.770949720670391
# 컬럼 삭제 후 acc : 
# [0.06293319 0.31729789 0.24903294 0.05000574 0.02168197 0.26767615 0.03137212]

# 2. RandomForestClassifier
# 기존 acc : 0.8100558659217877
# 컬럼 삭제 후 acc :  
# [0.07060852 0.26544044 0.25912464 0.04894936 0.03873309 0.28097624 0.03616771]

# 3. GradientBoostingClassifier
# 기존 acc :  0.8547486033519553
# 컬럼 삭제 후 acc :  
# [0.11551963 0.48075184 0.12908022 0.04891729 0.00693149 0.18364595 0.03515358]

# 4. XGBClassifier
# 기존 acc :  0.8268156424581006
# 컬럼 삭제 후 acc : 
# [0.15412153 0.6105986  0.03736079 0.0729577  0.03766907 0.04485419  0.04243808]
#=========================================================================================================================#
