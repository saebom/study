import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.utils import all_estimators
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
    x, y, train_size=0.85, shuffle=True, random_state=2022
    )

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgorithms : ', allAlgorithms)  

for (name, algorithm) in allAlgorithms: 
    try :
        model = algorithm()
        scores = cross_val_score(model, x, y, 
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

