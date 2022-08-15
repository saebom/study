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

'''
# 그래프 및 상관관계 확인
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14,20))

sns.set_theme(style="white") 
cols=['ProdTaken','Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
    'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',
    'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome']
for i, variable in enumerate(cols):
                     plt.subplot(10,2,i+1)
                     order = all_data_set[variable].value_counts(ascending=False).index   
                     #sns.set_palette(list_palette[i]) # to set the palette
                     sns.set_palette('Set2')
                     ax=sns.countplot(x=all_data_set[variable], data=all_data_set )
                     sns.despine(top=True,right=True,left=True) # to remove side line from graph
                     for p in ax.patches:
                           percentage = '{:.1f}%'.format(100 * p.get_height()/len(all_data_set[variable]))
                           x = p.get_x() + p.get_width() / 2 - 0.05
                           y = p.get_y() + p.get_height()
                           plt.annotate(percentage, (x, y),ha='center')
                     plt.tight_layout()
                     plt.title(cols[i].upper())

sns.set_palette(sns.color_palette("Set2", 8))
plt.figure(figsize=(15,10))
sns.heatmap(all_data_set.corr(),annot=True)
plt.show()
'''

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

print(all_data_set['TypeofContact'].value_counts())
all_data_set['TypeofContact'].mode()
all_data_set['TypeofContact']=all_data_set['TypeofContact'].fillna('Self Enquiry')

###(2) DurationOfPitch 와 TypeofContact
all_data_set['DurationOfPitch'].isnull().sum()
all_data_set['DurationOfPitch'].value_counts()
# all_data_set['Age'] = all_data_set['Age'].fillna(all_data_set['Age'].median())
# all_data_set['MonthlyIncome'] = all_data_set['MonthlyIncome'].fillna(all_data_set['MonthlyIncome'].median())
all_data_set['DurationOfPitch'] = all_data_set['DurationOfPitch'].fillna(all_data_set['DurationOfPitch'].median())
all_data_set['TypeofContact'].value_counts()

# all_data_set.Gender = all_data_set.Gender.replace("Fe Male","Female")
all_data_set['Gender'] = all_data_set['Gender'].apply(lambda x: 'Female' if x == 'Fe Male' else x)

# Age 와 MonthlyIncome
all_data_set.groupby(["Designation", "Gender","MaritalStatus"])["Age"].median()
all_data_set["Age"] = all_data_set.groupby(["Designation", "Gender","MaritalStatus"])["Age"].apply(
    lambda x: x.fillna(x.median())
)
all_data_set.groupby(["Occupation",'Designation','Gender'])["MonthlyIncome"].median()
all_data_set["MonthlyIncome"]=all_data_set.groupby(["Occupation",'Designation','Gender'])["MonthlyIncome"].apply(
    lambda x: x.fillna(x.median())
)

print(all_data_set.sort_values(by=["MonthlyIncome"],ascending = False).head(5))

all_data_set.fillna(all_data_set[all_data_set.DurationOfPitch>37].median())
all_data_set.fillna(all_data_set[(all_data_set.MonthlyIncome>40000) | (all_data_set.MonthlyIncome<12000)].median())
all_data_set.fillna(all_data_set[all_data_set.NumberOfTrips>10].median())

print(all_data_set.info())

for col in ['NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting']:
# for col in ['NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting']:
    # all_data_set[col] = all_data_set[col].fillna(all_data_set[col].mode()[0])
    all_data_set[col] = all_data_set[col].fillna(all_data_set[col].median())
    
print(all_data_set.info())

# all_data_set = all_data_set.drop(['PitchSatisfactionScore','ProductPitched','NumberOfFollowups','DurationOfPitch'],axis=1)

# x, y 데이터
# all_data_set을 train_set과 test_set으로 분할
train_set = all_data_set[:len(train_set)]
test_set = all_data_set[len(train_set):]
print(train_set.shape, test_set.shape)  # (1955, 14) (2933, 14)

x = train_set
y = label   # train_set['ProdTaken']

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123, stratify=y
    )

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = {'n_estimators': [100],
              'learning_rate' : [0.3],
              'max_depth' : [10], 
              'gamma': [1],
              'alpha' : [0.5],
            #   'min_child_weight': [1,5],
            #   'subsample' : [0.7,1],
            #   'colsample_bytree' : [0.7,1],
            #   'colsample_bylevel' : [0.7,1],
            #   'colsample_bynode' : [0.7,1],
            #   'reg_alpha' : [0, 0.1],
            #   'reg_lambda' : [0, 0.1],
              } 


#2. 모델구성
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# model = RandomForestClassifier()
# xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
model = XGBClassifier(learning_rate=0.3, max_depth=10, random_state=72, 
                      tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
# model = GridSearchCV(xgb, parameters, verbose=2, cv=kfold, n_jobs=8)


#3. 훈련
model.fit(x_train, y_train) 


#4. 평가, 예측
result = model.score(x_test, y_test)    
print('model.score : ', result) 
# print(model.best_params_)

y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape)   # (2933,)

# submission summit
submission['ProdTaken'] = y_summit
print(submission)
submission.to_csv('./_data/travel/submission5.csv', index=False)


#================================= 결과 ====================================#
# GridSearch 적용 전 model.score :  0.8900255754475703
# GridSearch 적용 후 model.score :  0.8797953964194374
# 08.13 model.score :  0.8925831202046036
# 08.15 model.score :  0.8951406649616368
#===========================================================================#