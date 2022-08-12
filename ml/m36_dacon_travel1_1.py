import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = './_data/travel/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'travel_submission.csv')

print('train.shape, test.shape, submit.shape', 
       train_set.shape, test_set.shape, submission.shape)    # (1955, 19) (2933, 18) (2933, 2)


# 'ProdTaken'과  'MonthlyIncome'의 관계
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.scatter(x = train_set['ProdTaken'], y = train_set['MonthlyIncome'])
# plt.ylabel('ProdTaken', fontsize = 13)
# plt.ylabel('MonthlyIncome', fontsize = 13)
# plt.show()

# 이상치 제거 'ProdTaken'와  'MonthlyIncome'
# train_set = train_set.drop(train_set[(train_set['MonthlyIncome']>80000) 
#                                      & (train_set['ProdTaken']>=0)].index) 
# 데이터 전처리
train_set[['TypeofContact', 'ProdTaken']].groupby(['TypeofContact'], 
                                          as_index=False).mean().sort_values(by='ProdTaken', ascending=False)
train_set[['Occupation', 'ProdTaken']].groupby(['Occupation'], 
                                       as_index=False).mean().sort_values(by='ProdTaken', ascending=False)
train_set[['Gender', 'ProdTaken']].groupby(['Gender'], 
                                         as_index=False).mean().sort_values(by='ProdTaken', ascending=False)
train_set[['MaritalStatus', 'ProdTaken']].groupby(['MaritalStatus'], 
                                         as_index=False).mean().sort_values(by='ProdTaken', ascending=False)
train_set[['Designation', 'ProdTaken']].groupby(['Designation'], 
                                         as_index=False).mean().sort_values(by='ProdTaken', ascending=False)

# all_data_set 데이터
label = train_set['ProdTaken']
all_data_set = pd.concat((train_set, test_set)).reset_index(drop=True)
all_data_set = all_data_set.drop(['ProdTaken'], axis=1)
print(all_data_set.shape)   # (4888, 18)
print(all_data_set.info())
print(all_data_set.describe())
print(all_data_set.columns) # Index(['Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
                            #    'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                            #    'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',
                            #    'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
                            #    'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'],
                            #     dtype='object')

# 결측값 조회
print(all_data_set.isnull().sum()) # Age 226, MonthlyIncome 233,
                                   # DurationOfPitch 251, TypeofContact 25, 
                                   # NumberOfFollowups 45, PreferredPropertyStar 26, 
                                   # NumberOfTrips 140, NumberOfChildrenVisiting 66
                                   
# 라벨인코딩
from sklearn.preprocessing import LabelEncoder
cols = ('TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation')
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data_set[c].values)) 
    all_data_set[c] = lbl.transform(list(all_data_set[c].values))
print(all_data_set.info())

    
# ###(1) Age 와 MonthlyIncome
# all_data_set['Age'] = all_data_set['Age'].fillna('None')
# all_data_set['MonthlyIncome'] = all_data_set['MonthlyIncome'].fillna('None')

###(2) DurationOfPitch 와 TypeofContact
all_data_set['DurationOfPitch'].isnull().sum()
all_data_set['DurationOfPitch'].value_counts()
# all_data_set['Age'] = all_data_set['Age'].fillna(all_data_set['Age'].median())
# all_data_set['MonthlyIncome'] = all_data_set['MonthlyIncome'].fillna(all_data_set['MonthlyIncome'].median())
all_data_set['DurationOfPitch'] = all_data_set['DurationOfPitch'].fillna(all_data_set['DurationOfPitch'].median())
all_data_set['TypeofContact'].value_counts()
for col in ['Age', 'MonthlyIncome', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting']:
# for col in ['NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting']:
    # all_data_set[col] = all_data_set[col].fillna(all_data_set[col].mode()[0])
    all_data_set[col] = all_data_set[col].fillna(all_data_set[col].median())

def outliers(df, col):
    out = []
    m = np.mean(df[col])
    sd = np.std(df[col])
    
    for i in df[col]: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
            
    print("Outliers:", out)
    print("min",np.median(out))
    return np.median(out)
    
col = "MonthlyIncome"
medOutlier = outliers(train_set,col)
train_set[train_set[col] >= medOutlier]

col = "NumberOfTrips"
medOutlier = outliers(train_set,col)
train_set[train_set[col] >= medOutlier]

                                
# x, y 데이터
# all_data_set을 train_set과 test_set으로 분할
train_set = all_data_set[:len(train_set)]
test_set = all_data_set[len(train_set):]
print(train_set.shape, test_set.shape)  # (1955, 18) (2933, 18)

x = train_set
y = label   # train_set['ProdTaken']

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

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=72)

#2. 모델구성
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
model = XGBClassifier(random_state=123, 
                      n_estimators=1000, 
                      learning_rate = 0.8,
                      max_depth = 10, 
                      gamma= 1,
                    )

# model = GridSearchCV(xgb, parameters, verbose=2, cv=kfold, n_jobs=8)

#3. 훈련
model.fit(x_train, y_train, early_stopping_rounds=10,
          eval_set = [(x_train, y_train), (x_test, y_test)],
        #   eval_set = [(x_test, y_test)], #eval_set은 tensorflow의 metrics와 동일
          eval_metric='error',
        #  회귀 : rmse, mae, rmsle...
        #  이진 : error, auc..., logloss...
        #  다중 : merror, mlogloss... 
          )


#4. 평가, 예측
result = model.score(x_test, y_test)    
print('model.score : ', result) 

y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape)   # (2933,)

# submission summit
submission['ProdTaken'] = y_summit
print(submission)
submission.to_csv('./_data/travel/submission2.csv', index=False)


#================================= 결과 ====================================#
# GridSearch 적용 전 model.score :  0.8900255754475703
# GridSearch 적용 후 model.score :  0.8849104859335039
#===========================================================================#