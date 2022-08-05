import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=66
# )

n_splite = 5
kfold = KFold(n_splits=n_splite, shuffle=True, random_state=66)


#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms: 
    try :
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold)
        print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))
    except:
        # continue
        print(name, '은 안나온 놈!!!')
        


#===================================== 결  과 ==========================================#
# ACC :  [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133] 
#  cross_val_score :  0.9649
# ACC :  [0.93859649 0.92982456 0.95614035 0.93859649 0.97345133] 
#  cross_val_score :  0.9473
# ACC :  [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 
#  cross_val_score :  0.6274
# ACC :  [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133] 
#  cross_val_score :  0.9263
# ACC :  [nan nan nan nan nan] 
#  cross_val_score :  nan
# ClassifierChain 은 안나온 놈!!!
# ACC :  [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531]
#  cross_val_score :  0.8963
# ACC :  [0.9122807  0.9122807  0.92982456 0.92105263 0.94690265] 
#  cross_val_score :  0.9245
# ACC :  [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858]
#  cross_val_score :  0.6274
# ACC :  [0.94736842 0.92982456 0.92982456 0.94736842 0.92035398] 
#  cross_val_score :  0.9349
# ACC :  [0.96491228 0.98245614 0.96491228 0.95614035 0.98230088] 
#  cross_val_score :  0.9701
# ACC :  [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221]
#  cross_val_score :  0.942
# ACC :  [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265] 
#  cross_val_score :  0.9122
# ACC :  [0.95614035 0.96491228 0.95614035 0.93859649 0.97345133] 
#  cross_val_score :  0.9578
# ACC :  [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088] 
#  cross_val_score :  0.9737
# ACC :  [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 
#  cross_val_score :  0.928
# ACC :  [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 
#  cross_val_score :  0.3902
# ACC :  [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 
#  cross_val_score :  0.3902
# ACC :  [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133] 
#  cross_val_score :  0.9614
# ACC :  [0.92105263 0.88596491 0.9122807  0.87719298 0.94690265] 
#  cross_val_score :  0.9087
# ACC :  [0.93859649 0.95614035 0.88596491 0.95614035 0.94690265] 
#  cross_val_score :  0.9367
# ACC :  [0.95614035 0.97368421 0.9122807  0.96491228 0.96460177] 
#  cross_val_score :  0.9543
# ACC :  [0.88596491 0.94736842 0.92105263 0.94736842 0.94690265] 
#  cross_val_score :  0.9297
# MultiOutputClassifier 은 안나온 놈!!!
# ACC :  [0.85964912 0.92105263 0.87719298 0.9122807  0.89380531]
#  cross_val_score :  0.8928
# ACC :  [0.86842105 0.89473684 0.85964912 0.9122807  0.91150442] 
#  cross_val_score :  0.8893
# ACC :  [0.85964912 0.9122807  0.83333333 0.87719298 0.88495575] 
#  cross_val_score :  0.8735
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# ACC :  [0.90350877 0.93859649 0.75438596 0.8245614  0.90265487] 
#  cross_val_score :  0.8647
# ACC :  [0.40350877 0.80701754 0.85964912 0.86842105 0.94690265] 
#  cross_val_score :  0.7771
# ACC :  [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265] 
#  cross_val_score :  0.9525
# ACC :  [nan nan nan nan nan] 
#  cross_val_score :  nan
# ACC :  [0.96491228 0.96491228 0.96491228 0.96491228 0.98230088] 
#  cross_val_score :  0.9684
# ACC :  [0.95614035 0.98245614 0.92105263 0.95614035 0.95575221] 
#  cross_val_score :  0.9543
# ACC :  [0.94736842 0.97368421 0.93859649 0.95614035 0.96460177] 
#  cross_val_score :  0.9561
# ACC :  [0.83333333 0.88596491 0.87719298 0.60526316 0.92920354] 
#  cross_val_score :  0.8262
# ACC :  [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 
#  cross_val_score :  0.921
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
#=======================================================================================#

