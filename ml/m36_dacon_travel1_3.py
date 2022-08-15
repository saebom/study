import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
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


# 오타
all_data_set.Gender = all_data_set.Gender.replace("Fe Male","Female")
print(all_data_set.Gender.value_counts())

# outlier
outlier_num = all_data_set.select_dtypes(include=np.number)
Q1 = all_data_set.quantile(0.25)            
Q3 = all_data_set.quantile(0.75)

IQR = Q3 - Q1                           

lower=Q1-1.5*IQR                        
upper=Q3+1.5*IQR

print(((outlier_num<lower)|(outlier_num>upper)).sum()/len(all_data_set)*100)

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
    
cols = ["MonthlyIncome", "NumberOfTrips"]
for col in cols :
    medOutlier = outliers(all_data_set,col)
    all_data_set[all_data_set[col] >= medOutlier]
    print(medOutlier)
    
# 결측치 보간
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
imp = IterativeImputer(estimator = LinearRegression(), 
                       tol= 1e-10, 
                       max_iter=100, 
                       verbose=2, 
                       imputation_order='ascending')

all_data_set = pd.DataFrame(imp.fit_transform(all_data_set))
print(all_data_set.info())

# outlier
outlier_num = all_data_set.select_dtypes(include=np.number)
Q1 = all_data_set.quantile(0.25)            
Q3 = all_data_set.quantile(0.75)

IQR = Q3 - Q1                           

lower=Q1-1.5*IQR                        
upper=Q3+1.5*IQR
print(((outlier_num<lower)|(outlier_num>upper)).sum()/len(all_data_set)*100)

# x, y 데이터
# all_data_set을 train_set과 test_set으로 분할
train_set = all_data_set[:len(train_set)]
test_set = all_data_set[len(train_set):]
print(train_set.shape, test_set.shape)  # (1955, 18) (2933, 18)

x = train_set
y = label   # train_set['ProdTaken']

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72
    )

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# model = RandomForestClassifier()
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
# model = XGBClassifier(random_state=123, 
#                       n_estimators=250, 
#                       learning_rate = 0.3,
#                       max_depth = 10, 
#                       min_child_weight=1,
#                       gamma=0,
#                       subsample=0.8,
#                       colsample_bytree=0.8,
#                       objective= 'binary:logistic',
#                       nthread=-1,
#                       scale_pos_weight=1,
#                       max_bin=256,
#                       max_cat_to_onehot=4
#                       )
# model = GridSearchCV(xgb, parameters, verbose=2, cv=kfold, n_jobs=8)

#3. 훈련
model.fit(x_train, y_train) 


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
# GridSearch 적용 후 model.score :  0.8797953964194374
# 0.8925831202046036
#===========================================================================#