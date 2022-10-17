import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

import random
import os
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(123) # Seed 고정


#1. 데이터
path = './_data/autodriving/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'submission.csv')

x_train = train_set.filter(regex='X')   # Input : X Featrue : 56
y_train = train_set.filter(regex='Y')   # Output : Y Feature : 14
x_test = test_set.drop(columns=['ID'])
# x_test = test_set.filter(regex='X')
# y_test = test_set.filter(regex='Y')

# print('x_train.shape, y_train.shape, x_test.shape', 
#        train_set.shape, test_set.shape, submission.shape)    # (39607, 70) (39608, 56) (39608, 15)
 
print('x_train.shape, y_train.shape, x_test.shape',
        x_train.shape, y_train.shape, x_test.shape) # (39607, 56) (39607, 14) (39608, 56)

col = ["X_10","X_11"]
# col = ["X_01","X_02", "X_03","X_04", "X_05","X_06", "X_07", "X_08", "X_09","X_10","X_11",
#         "X_12", "X_13","X_14", "X_15","X_16", "X_17", "X_18", "X_19","X_20",]
# col = ["X_01","X_02", "X_03","X_04", "X_05","X_06", "X_07","X_08", "X_09", "X_10","X_11","X_12",
#         "X_13","X_14", "X_15","X_16", "X_17", "X_18", "X_19", "X_20","X_21","X_22", "X_23","X_24", "X_25","X_26", 
#         "X_27","X_28", "X_29", "X_30", "X_31", "X_32","X_33","X_34", "X_35","X_36", "X_37","X_38",
#         "X_39","X_40", "X_41","X_46", "X_49","X_50", "X_51", "X_52","X_53","X_54","X_55"]
x_train[col] = x_train[col].replace(0, np.nan)

# MICE 결측치 보간
imp = IterativeImputer(estimator = LinearRegression(), 
                       tol= 1e-10, 
                       max_iter=100, 
                       verbose=2, 
                       imputation_order='descending')

x_train = pd.DataFrame(imp.fit_transform(x_train))
# print(x_train)

# 이상치 확인
outlier_num = x_train.select_dtypes(include=np.number)
 
Q1 = x_train.quantile(0.25)            
Q3 = x_train.quantile(0.75)

IQR = Q3 - Q1                           

lower=Q1-1.5*IQR                        
upper=Q3+1.5*IQR

print(((outlier_num<lower)|(outlier_num>upper)).sum()/len(x_train)*100)


#2. 모델 구성
# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')
# print('allAlgorithms : ', allAlgorithms)  

for (name, algorithm) in allAlgorithms: 
    try :
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, 
                                #  cv=kfold
                                 )
        print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))
    except:
        # continue
        print(name, '은 안나온 놈!!!')
        


#===================================== 결  과 ==========================================#
# 'AdaBoostClassifier' ACC :  [0.8516624  0.83120205 0.83887468 0.83120205 0.82352941] 
#  cross_val_score :  0.8353
# 'BaggingClassifier'ACC :  [0.86700767 0.85933504 0.8516624  0.86189258 0.87468031] 
#  cross_val_score :  0.8629
# 'BernoulliNB' ACC :  [0.82352941 0.84143223 0.84654731 0.82608696 0.82352941] 
#  cross_val_score :  0.8322
# 'CalibratedClassifierCV' ACC :  [0.8056266  0.8056266  0.80051151 0.80306905 0.80306905] 
#  cross_val_score :  0.8036
# 'CategoricalNB' ACC :  [       nan 0.84398977        nan        nan 0.84910486] 
#  cross_val_score :  nan
# ClassifierChain 은 안나온 놈!!!
# 'ComplementNB' ACC :  [0.66496164 0.64705882 0.60102302 0.62404092 0.62148338] 
#  cross_val_score :  0.6317
# 'DecisionTreeClassifier' ACC :  [0.84654731 0.83631714 0.80818414 0.84143223 0.84398977] 
#  cross_val_score :  0.8353
# 'DummyClassifier' ACC :  [0.8056266  0.8056266  0.80306905 0.80306905 0.80306905]
#  cross_val_score :  0.8041
# 'ExtraTreeClassifier' ACC :  [0.76982097 0.83887468 0.7826087  0.77493606 0.76982097] 
#  cross_val_score :  0.7872
# 'ExtraTreesClassifier' ACC :  [0.91304348 0.89002558 0.87723785 0.88746803 0.88491049] 
#  cross_val_score :  0.8905
# 'GaussianNB' ACC :  [0.84143223 0.83375959 0.81074169 0.81329923 0.8286445 ]
#  cross_val_score :  0.8256
# 'GaussianProcessClassifier' ACC :  [0.74936061 0.78005115 0.76470588 0.73401535 0.78516624] 
#  cross_val_score :  0.7627
# 'GradientBoostingClassifier' ACC :  [0.86956522 0.86189258 0.84143223 0.86700767 0.84910486] 
#  cross_val_score :  0.8578
# 'HistGradientBoostingClassifier' ACC :  [0.89258312 0.88491049 0.87212276 0.86445013 0.87468031] 
#  cross_val_score :  0.8777
# 'KNeighborsClassifier' ACC :  [0.74680307 0.79028133 0.76982097 0.7314578  0.77237852] 
#  cross_val_score :  0.7621
# 'LabelPropagation'ACC :  [0.83631714 0.85421995 0.83375959 0.8286445  0.83375959] 
#  cross_val_score :  0.8373
# 'LabelSpreading' ACC :  [0.83631714 0.85421995 0.83375959 0.8286445  0.83375959] 
#  cross_val_score :  0.8373
# 'LinearDiscriminantAnalysis' ACC :  [0.84910486 0.8286445  0.84654731 0.82097187 0.80818414] 
#  cross_val_score :  0.8307
# 'LinearSVC' ACC :  [0.8056266  0.8056266  0.19693095 0.80306905 0.19693095] 
#  cross_val_score :  0.5616
# 'LogisticRegression' ACC :  [0.83375959 0.82352941 0.83375959 0.83375959 0.81329923] 
#  cross_val_score :  0.8276
# 'LogisticRegressionCV' ACC :  [0.81585678 0.80818414 0.83375959 0.80818414 0.81074169] 
#  cross_val_score :  0.8153
# 'MLPClassifier' ACC :  [0.8056266  0.1943734  0.80306905 0.19693095 0.19693095] 
#  cross_val_score :  0.4394
# MultiOutputClassifier 은 안나온 놈!!!
# 'MultinomialNB' ACC :  [0.76726343 0.75703325 0.72890026 0.74424552 0.75191816]
#  cross_val_score :  0.7499
# 'NearestCentroid' ACC :  [0.56521739 0.53964194 0.49104859 0.46803069 0.50127877] 
#  cross_val_score :  0.513
# 'NuSVC' ACC :  [nan nan nan nan nan] 
#  cross_val_score :  nan
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# 'PassiveAggressiveClassifier' ACC :  [0.8056266  0.8056266  0.19693095 0.19693095 0.80306905] 
#  cross_val_score :  0.5616
# 'Perceptron'ACC :  [0.8056266  0.1943734  0.80306905 0.80306905 0.80306905] 
#  cross_val_score :  0.6818
# 'QuadraticDiscriminantAnalysis' ACC :  [0.81074169 0.83120205 0.78772379 0.77493606 0.73401535] 
#  cross_val_score :  0.7877
# 'RadiusNeighborsClassifier' ACC :  [nan nan nan nan nan] 
#  cross_val_score :  nan
# RandomForestClassifier' ACC :  [0.87212276 0.86956522 0.87212276 0.87212276 0.87723785] 
#  cross_val_score :  0.8726
# 'RidgeClassifier' ACC :  [0.84143223 0.83631714 0.83887468 0.82097187 0.8056266 ] 
#  cross_val_score :  0.8286
# 'RidgeClassifierCV' ACC :  [0.84143223 0.83887468 0.83375959 0.82097187 0.81074169] 
#  cross_val_score :  0.8292
# 'SGDClassifier' ACC :  [0.1943734  0.8056266  0.80306905 0.80306905 0.80306905] 
#  cross_val_score :  0.6818
# 'SVC' ACC :  [0.8056266  0.8056266  0.80306905 0.80306905 0.80306905] 
#  cross_val_score :  0.8041
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
#=======================================================================================#

