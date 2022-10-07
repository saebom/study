import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = './_data/travel/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'travel_submission.csv')

print('train.shape, test.shape, submit.shape', 
       train_set.shape, test_set.shape, submission.shape)    # (1955, 19) (2933, 18) (2933, 2)

# train_set 데이터
for i in train_set.columns:
    print(i,"column's unique values are:",train_set[i].unique())
print(train_set.shape)   # (4888, 18)
print(train_set.info())
print(train_set.describe())
print(train_set.columns) 
# Index(['Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
# 'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
# 'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',
# 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
# 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'],
#  dtype='object')

# 결측값 조회
print(train_set.isnull().sum()) 
# Age 226, MonthlyIncome 233,
# DurationOfPitch 251, TypeofContact 25, 
# NumberOfFollowups 45, PreferredPropertyStar 26, 
# NumberOfTrips 140, NumberOfChildrenVisiting 66

# TypeofContact 전처리
train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
train_set['Age']=np.round(train_set['Age'],0).astype(int)
test_set['Age']=np.round(test_set['Age'],0).astype(int)

train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
print(train_set.describe) #(1955, 19)
print(train_set[train_set['MonthlyIncome'].notnull()].groupby(['Designation'])['MonthlyIncome'].mean())

train_set['NumberOfChildrenVisiting'].fillna(train_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
test_set['NumberOfChildrenVisiting'].fillna(test_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
train_set['NumberOfFollowups'].fillna(train_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
test_set['NumberOfFollowups'].fillna(test_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)

train_set['DurationOfPitch']=train_set['DurationOfPitch'].fillna(0)
test_set['DurationOfPitch']=test_set['DurationOfPitch'].fillna(0)

print(train_set[train_set['DurationOfPitch'].notnull()].groupby(['NumberOfChildrenVisiting'])['DurationOfPitch'].mean())


train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)

combine = [train_set,test_set]
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 20, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 29), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 59), 'Age'] = 4
    dataset.loc[ dataset['Age'] > 59, 'Age'] = 5

train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)

train_set.loc[ train_set['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'
test_set.loc[ test_set['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'

train_set.loc[ train_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'
test_set.loc[ test_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'

                                     
# 라벨인코딩
from sklearn.preprocessing import LabelEncoder
cols = ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train_set[c].values)) 
    lbl.fit(list(test_set[c].values)) 
    train_set[c] = lbl.transform(list(train_set[c].values))
    test_set[c] = lbl.transform(list(test_set[c].values))
print(train_set.info())

# MonthlyIncome 이상치 확인
print(train_set.sort_values(by=["MonthlyIncome"],ascending = False).head(5))
print(train_set[(train_set.MonthlyIncome>40000) | (train_set.MonthlyIncome<12000)])

# NumberOfTrips 이상치 확인
print(train_set.sort_values(by=["NumberOfTrips"],ascending = False).head(5))

# DurationOfPitch 이상치 제거
DurationOfPitch_out_index = train_set.drop(index=train_set[train_set.DurationOfPitch>37].index,inplace=True)

# MonthlyIncome 이상치 제거
# MonthlyIncome_out_index = train_set.drop(index=train_set[(train_set.MonthlyIncome>40000) | (train_set.MonthlyIncome<12000)].index,inplace=True)

# NumberOfTrips 이상치 제거
NumberOfTrips_out_index = train_set.drop(index=train_set[train_set.NumberOfTrips>10].index,inplace=True)
 
x = train_set.drop(['ProdTaken',
                          'NumberOfChildrenVisiting',
                          'NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups',
                        #   'TypeofContact',
                          ], axis=1)
# x = train_set.drop(['ProdTaken'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting',
                          'NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups',
                        #   'TypeofContact',
                          ], axis=1)
y = train_set['ProdTaken']
print(x.shape) #1911,13


from sklearn.model_selection import train_test_split, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, shuffle=True, random_state=1234, stratify=y
    )

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=123)

cat_paramets = {"learning_rate" : [0.01],
                'depth' : [8],
                'od_pval' : [0.12673190617341812],
                'fold_permutation_block': [142],
                'l2_leaf_reg' :[0.33021257848638497]}

#2. 모델구성
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

cat = CatBoostClassifier(random_state=1127,verbose=False,n_estimators=1304)
model = RandomizedSearchCV(cat,cat_paramets,cv=kfold,n_jobs=-1,)

import time 
start_time = time.time()
model.fit(x_train,y_train)   
end_time = time.time()-start_time 
y_predict = model.predict(x_test)

results = accuracy_score(y_test,y_predict)
print('최적의 매개변수 : ',model.best_params_)
print('최상의 점수 : ',model.best_score_)
print('acc :',results)
print('f1_score(macro) : ', f1_score(y_test, y_predict))   
print('시간 : ', end_time)

model.fit(x,y)
y_summit = model.predict(test_set)
y_summit = np.round(y_summit,0)
print(y_summit)
print(y_summit.shape)   # (2933,) 

# submission summit
submission['ProdTaken'] = y_summit
print(submission)
submission.to_csv('./_data/travel/submission_0902_2.csv', index=False)


#================================= 결과 ====================================#
# submission_0902_1.csv (0.9249786871) id: spring22
# 최상의 점수 :  0.9036144578313253
# acc : 0.9455782312925171
# f1_score(macro) :  0.8367346938775511
#===========================================================================#
