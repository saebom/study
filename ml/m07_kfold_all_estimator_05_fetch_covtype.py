<<<<<<< HEAD
import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=72
# )

n_splits =5
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
# ACC :  [0.55381531 0.41932652 0
#  cross_val_score ;  0.5184
# ACC :  [0.96207499 0.96118861 0
#  cross_val_score ;  0.9616
# ACC :  [0.63382185 0.63033657 0
#  cross_val_score ;  0.6313
# ACC :  [0.6722718  0.68255553 0
#  cross_val_score ;  0.6815
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# ClassifierChain 은 안나온 놈!!!
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# ACC :  [0.93867628 0.93948521 0
#  cross_val_score ;  0.9394
# ACC :  [0.48858463 0.48664837 0
#  cross_val_score ;  0.4876
# ACC :  [0.857835   0.86207757 0
#  cross_val_score ;  0.8617
# ACC :  [0.9542955  0.95308211 0
#  cross_val_score ;  0.9534
# ACC :  [0.4565889  0.45806046 0
#  cross_val_score ;  0.459
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# ACC :  [0.77273392 0.7717787  0
#  cross_val_score ;  0.7727
# ACC :  [0.77929141 0.78459248 0
#  cross_val_score ;  0.7881
# ACC :  [0.96880459 0.9688132  0
#  cross_val_score ;  0.9688
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# ACC :  [0.68061066 0.67919073 0
#  cross_val_score ;  0.6798
# ACC :  [0.53957299 0.61398587 0
#  cross_val_score ;  0.5788
# ACC :  [0.61996678 0.62153301 0
#  cross_val_score ;  0.6212
# ACC :  [0.67208248 0.66703097 0
#  cross_val_score ;  0.6692
# ACC :  [0.76649484 0.76209736 0
#  cross_val_score ;  0.7604
# MultiOutputClassifier 은 안나온
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# ACC :  [0.19400532 0.19407416 0
#  cross_val_score ;  0.1939
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 
# ACC :  [0.37369947 0.55767923 0
# ACC :  [0.71485246 0.71494712 0.71597735 0.71472092 0.71213921]
#  cross_val_score ;  0.7145
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
#=======================================================================================#

import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=72
# )

n_splits =5
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
# (tf282gpu) C:\Users\bitcamp\DocFiles\lib\python\debugpy\adapte
# ACC :  [0.55381531 0.41932652 0
#  cross_val_score ;  0.5184
# ACC :  [0.96207499 0.96118861 0
#  cross_val_score ;  0.9616
# ACC :  [0.63382185 0.63033657 0
#  cross_val_score ;  0.6313
# ACC :  [0.6722718  0.68255553 0
#  cross_val_score ;  0.6815
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# ClassifierChain 은 안나온 놈!!!
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# ACC :  [0.93867628 0.93948521 0
#  cross_val_score ;  0.9394
# ACC :  [0.48858463 0.48664837 0
#  cross_val_score ;  0.4876
# ACC :  [0.857835   0.86207757 0
#  cross_val_score ;  0.8617
# ACC :  [0.9542955  0.95308211 0
#  cross_val_score ;  0.9534
# ACC :  [0.4565889  0.45806046 0
#  cross_val_score ;  0.459
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# ACC :  [0.77273392 0.7717787  0
#  cross_val_score ;  0.7727
# ACC :  [0.77929141 0.78459248 0
#  cross_val_score ;  0.7881
# ACC :  [0.96880459 0.9688132  0
#  cross_val_score ;  0.9688
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# ACC :  [0.68061066 0.67919073 0
#  cross_val_score ;  0.6798
# ACC :  [0.53957299 0.61398587 0
#  cross_val_score ;  0.5788
# ACC :  [0.61996678 0.62153301 0
#  cross_val_score ;  0.6212
# ACC :  [0.67208248 0.66703097 0
#  cross_val_score ;  0.6692
# ACC :  [0.76649484 0.76209736 0
#  cross_val_score ;  0.7604
# MultiOutputClassifier 은 안나온
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# ACC :  [0.19400532 0.19407416 0
#  cross_val_score ;  0.1939
# ACC :  [nan nan nan nan nan]
#  cross_val_score ;  nan
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 
# ACC :  [0.37369947 0.55767923 0
# ACC :  [0.71485246 0.71494712 0.71597735 0.71472092 0.71213921]
#  cross_val_score ;  0.7145
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
#=======================================================================================#
