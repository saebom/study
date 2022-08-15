from xml.parsers.expat import model
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.random.set_seed(123)


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

    
# ###(1) Age 와 MonthlyIncome
# all_data_set['Age'] = all_data_set['Age'].fillna('None')
# all_data_set['MonthlyIncome'] = all_data_set['MonthlyIncome'].fillna('None')
all_data_set["MonthlyIncome"] = all_data_set.groupby(["Designation"])["MonthlyIncome"].transform(lambda x: x.fillna(x.median()))
all_data_set["Age"] = all_data_set.groupby(["Designation"])["Age"].transform(lambda x: x.fillna(x.median()))

# create list of numerical columns
missing_numerical = all_data_set.select_dtypes(include=np.number).columns.tolist()
missing_numerical.remove("MonthlyIncome")
missing_numerical.remove("Age")

# function for replacing with the Median value of the attributes
medianFiller = lambda x: x.fillna(x.median()) 

# apply the function
all_data_set[missing_numerical] = all_data_set[missing_numerical].apply(medianFiller,axis=0)

# treating missing values in remaining categorical variables
all_data_set["TypeofContact"] = all_data_set["TypeofContact"].fillna("Self Enquiry")
all_data_set["NumberOfChildrenVisiting"] = all_data_set["NumberOfChildrenVisiting"].fillna(1.0)
all_data_set["PreferredPropertyStar"] = all_data_set["PreferredPropertyStar"].fillna(3.0)

# 오타
all_data_set.Gender = all_data_set.Gender.replace("Fe Male","Female")

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

outlier_num = all_data_set.select_dtypes(include=np.number)
 
# find the 25th percentile and 75th percentile.
Q1 = all_data_set.quantile(0.25)            
Q3 = all_data_set.quantile(0.75)

# Inter Quantile Range (75th percentile - 25th percentile)
IQR = Q3 - Q1                           

# find lower and upper bounds for all values. All values outside these bounds are outliers
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
    
col = "MonthlyIncome"
medOutlier = outliers(all_data_set,col)
all_data_set[all_data_set[col] >= medOutlier]

col = "NumberOfTrips"
medOutlier = outliers(all_data_set,col)
all_data_set[all_data_set[col] >= medOutlier]

# all_data_set= all_data_set.drop(["PitchSatisfactionScore","ProductPitched","NumberOfFollowups","DurationOfPitch"],axis=1)

                                
# x, y 데이터
# all_data_set을 train_set과 test_set으로 분할
train_set = all_data_set[:len(train_set)]
test_set = all_data_set[len(train_set):]
print(train_set.shape, test_set.shape)  # (1955, 18) (2933, 18)

x = train_set
y = label   # train_set['ProdTaken']


from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=72, stratify=y
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# parameters = {'n_estimators': [100],
#               'learning_rate' : [0.1],
#               'max_depth' : [3], #default 6 => 통상 max는 4정도에서 성능이 좋다
#               'gamma': [1],
#               'min_child_weight': [1],
#               'subsample' : [1],
#               'colsample_bytree' : [1],
#               'colsample_bylevel' : [1],
#               'colsample_bynode' : [1],
#               'reg_alpha' : [0],
#               'reg_lambda' : [1],
#               }  


#2. 모델구성
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# model = RandomForestClassifier(n_estimators=100, random_state=123, max_depth=10, max_features=2)
# model = LogisticRegression(C=0.1)
# model = DecisionTreeClassifier(criterion="gini",class_weight={0:0.15,1:0.85},random_state=1)
# model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
# model = XGBClassifier(random_state=72)
# model = GridSearchCV(xgb, parameters, verbose=2, cv=kfold, n_jobs=8)
dtc1 = DecisionTreeClassifier(class_weight={0:0.15,1:0.85},random_state=1)
bgcdt = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion="gini",
                                                                class_weight={0:0.15,1:0.85},random_state=1),random_state=1)
bgc = BaggingClassifier(random_state=1)
bgclr = BaggingClassifier(base_estimator=LogisticRegression(solver="liblinear", random_state=1),random_state=1) 
bgcdt = BaggingClassifier(base_estimator=DecisionTreeClassifier(
    criterion="gini",class_weight={0:0.15,1:0.85},random_state=1),random_state=1)
rfc = RandomForestClassifier(random_state=1)
rfc1 = RandomForestClassifier(class_weight={0:0.15,1:0.85},random_state=1)
xgbc = XGBClassifier(random_state=1, eval_metric="logloss")
xgbc1 = XGBClassifier(random_state=123, 
                      n_estimators=250, 
                      learning_rate = 0.3,
                      max_depth = 10, 
                      min_child_weight=1,
                      gamma=0,
                      subsample=0.8,
                      colsample_bytree=0.8,
                      objective= 'binary:logistic',
                      nthread=-1,
                      scale_pos_weight=1,
                      max_bin=256,
                      max_cat_to_onehot=4
                      )

# parameters = {"max_depth": np.arange(10,60,10), 
#             "criterion": ["gini","entropy"],
#             "min_samples_leaf": [ 2, 5, 7, 10],
#             "max_leaf_nodes" : [3, 5, 10,15],}
parameters = { "max_depth":[3,10,15],
            #    "n_estimators": [150,200,250,500],
            #    'min_child_weight':range(1,6,2),
              }
# parameters = {
#     "n_estimators": np.arange(10,60,10),
#     "subsample":[0.6,0.7,0.8],
#     "learning_rate":[0.1,0.3,0.55],
#     "colsample_bytree":[0.5,0.7,0.9],
#     "colsample_bylevel":[0.5,0.7,0.9]
# }
# type of scoring used to compare parameter combinations
from sklearn import metrics
scorer = metrics.make_scorer(metrics.f1_score)
grid_obj = GridSearchCV(xgbc1, parameters, scoring=scorer,cv=5, verbose=2)
grid_obj = grid_obj.fit(x_train, y_train)
xgbc1 = grid_obj.best_estimator_
print('최고의 파라미터 : ', xgbc1)

# verify the update
print(all_data_set.Gender.value_counts())

def get_metrics_score(model,flag=True):

    # defining an empty list to store train and test results
    score_list=[] 
    
    # predicting on train and tests
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    
    # accuracy of the model
    train_acc = model.score(x_train,y_train)
    test_acc = model.score(x_test,y_test)
    
    # recall of the model
    train_recall = metrics.recall_score(y_train,pred_train)
    test_recall = metrics.recall_score(y_test,pred_test)
    
    # precision of the model
    train_precision = metrics.precision_score(y_train,pred_train)
    test_precision = metrics.precision_score(y_test,pred_test)
    
    # f1_score of the model
    train_f1 = metrics.f1_score(y_train,pred_train)
    test_f1 = metrics.f1_score(y_test,pred_test)

    # populate the score_list 
    score_list.extend((train_acc,test_acc,train_recall,test_recall,train_precision,test_precision,train_f1,test_f1))
        
    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True: 
        print("Accuracy on training set : ",train_acc)
        print("Accuracy on test set : ",test_acc)
        print("Recall on training set : ",train_recall)
        print("Recall on test set : ",test_recall)
        print("Precision on training set : ",train_precision)
        print("Precision on test set : ",test_precision)
        print("F1 on training set : ",train_f1)
        print("F1 on test set : ",test_f1)
    return score_list # returning the list with train and test scores    
    
#3. 훈련
# model.fit(x_train, y_train) 

xgbc1.fit(x_train,y_train)
# bgcdt.fit(x_train,y_train)
xgbc1_score = get_metrics_score(xgbc1)
# rfc1.fit(x_train,y_train)


#4. 평가, 예측
result = xgbc1.score(x_test, y_test)    
print('model.score : ', result) 

# seeds = [123, 1234, 1, 0, 111, 555, 999, 666]
seeds =[123]
# seeds = [1979, 1999, 2000]

y_summit = xgbc1.predict(test_set)

for seed in seeds:
    xgbc1 = XGBClassifier(random_state=seed, 
                      n_estimators=250, 
                      learning_rate = 0.3,
                      max_depth = 10, 
                      min_child_weight=1,
                      gamma=0,
                      subsample=0.8,
                      colsample_bytree=0.8,
                      objective= 'binary:logistic',
                      nthread=-1,
                      scale_pos_weight=1,
                      max_bin=256,
                      max_cat_to_onehot=4
                      )
    xgbc1.fit(x_train,y_train)
    xgbc1_score = get_metrics_score(xgbc1)
    result = xgbc1.score(x_test, y_test)   
    print('model.score : ', result) 
    print('========================================') 
    
     
print('model.score : ', result) 
    
print(y_summit)
print(y_summit.shape)   # (2933,)

# submission summit
submission['ProdTaken'] = y_summit
print(submission)
submission.to_csv('./_data/travel/submission6.csv', index=False)


#================================= 결과 ====================================#
# GridSearch 적용 전 model.score :  0.8900255754475703
# GridSearch 적용 후 model.score :  0.8797953964194374
# 08.13 model.score :  0.8925831202046036
# 08.14 model.score :  0.8593350383631714
# 08.15 model.score :  0.8849104859335039
# ===============================================
# 08.15 submission4.csv:: 
# Accuracy on training set :  1.0
# Accuracy on test set :  0.9028132992327366
# Recall on training set :  1.0
# Recall on test set :  0.6493506493506493
# Precision on training set :  1.0
# Precision on test set :  0.819672131147541
# F1 on training set :  1.0
# F1 on test set :  0.7246376811594203
# model.score :  0.9028132992327366

# 08.15 submission5.csv:: 
# Accuracy on training set :  1.0
# Accuracy on test set :  0.9156010230179028
# Recall on training set :  1.0
# Recall on test set :  0.6883116883116883
# Precision on training set :  1.0
# Precision on test set :  0.8548387096774194
# F1 on training set :  1.0
# F1 on test set :  0.762589928057554
# model.score :  0.9156010230179028

# 최고의 파라미터 :  XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
#               colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.8,
#               early_stopping_rounds=None, enable_categorical=False,
#               eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
#               importance_type=None, interaction_constraints='',
#               learning_rate=0.3, max_bin=256, max_cat_to_onehot=4,
#               max_delta_step=0, max_depth=10, max_leaves=0, min_child_weight=1,
#               missing=nan, monotone_constraints='()', n_estimators=250,
#               n_jobs=-1, nthread=-1, num_parallel_tree=1, predictor='auto',
#               random_state=2019, reg_alpha=0, ...)

# 08.15 submission6.csv:: 
# Accuracy on training set :  1.0
# Accuracy on test set :  0.9156010230179028
# Recall on training set :  1.0
# Recall on test set :  0.6883116883116883
# Precision on training set :  1.0
# Precision on test set :  0.8548387096774194
# F1 on training set :  1.0
# F1 on test set :  0.762589928057554
# model.score :  0.9156010230179028
#===========================================================================#