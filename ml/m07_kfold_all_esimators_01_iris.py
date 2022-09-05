import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.utils import all_estimators, shuffle
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.random.set_seed(66)  # weight에 난수값 

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)   # 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'

x = datasets['data']
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#    x, y, train_size=0.8, shuffle=True, random_state=72
# )
n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)


#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

# print('allAlgorithms : ', allAlgorithms)    # allAlgorithms의 리스트 출력. key와 value로 이루어진 딕셔너리임.
print('모델의 갯수 : ', len(allAlgorithms))    # 41

for (name, algorithm) in allAlgorithms: 
    try :
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold)
        print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))
    except:
        # continue
        print(name, '은 안나온 놈!!!')



#===================================== 결  과 ==========================================#
# 모델의 갯수 :  41
#============================================================================================
# ACC :  [0.63333333 0.93333333 1.         0.9        0.96666667] 
#  cross_val_score :  0.8867
# ACC :  [0.96666667 0.96666667 1.         0.9        0.96666667] 
#  cross_val_score :  0.96
# ACC :  [0.3        0.33333333 0.3        0.23333333 0.3       ] 
#  cross_val_score :  0.2933
# ACC :  [0.9        0.83333333 1.         0.86666667 0.96666667] 
#  cross_val_score :  0.9133
# ACC :  [0.9        0.93333333 0.93333333 0.9        1.        ]
#  cross_val_score :  0.9333
# ClassifierChain 은 안나온 놈!!!
# ACC :  [0.66666667 0.66666667 0.7        0.6        0.7       ] 
#  cross_val_score :  0.6667
# ACC :  [0.96666667 0.96666667 1.         0.9        0.93333333]
#  cross_val_score :  0.9533
# ACC :  [0.3        0.33333333 0.3        0.23333333 0.3       ]
#  cross_val_score :  0.2933
# ACC :  [0.76666667 0.9        0.96666667 0.83333333 0.96666667] 
#  cross_val_score :  0.8867
# ACC :  [0.96666667 0.96666667 1.         0.86666667 0.96666667] 
#  cross_val_score :  0.9533
# ACC :  [0.96666667 0.9        1.         0.9        0.96666667]
#  cross_val_score :  0.9467
# ACC :  [0.96666667 0.96666667 1.         0.9        0.96666667] 
#  cross_val_score :  0.96
# ACC :  [0.93333333 0.96666667 1.         0.93333333 0.96666667] 
#  cross_val_score :  0.96
# ACC :  [0.86666667 0.96666667 1.         0.9        0.96666667] 
#  cross_val_score :  0.94
# ACC :  [0.96666667 0.96666667 1.         0.9        0.96666667] 
#  cross_val_score :  0.96
# ACC :  [0.93333333 1.         1.         0.9        0.96666667] 
#  cross_val_score :  0.96
# ACC :  [0.93333333 1.         1.         0.9        0.96666667] 
#  cross_val_score :  0.96
# ACC :  [1.  1.  1.  0.9 1. ] 
#  cross_val_score :  0.98
# ACC :  [0.96666667 0.96666667 1.         0.9        1.        ] 
#  cross_val_score :  0.9667
# ACC :  [1.         0.96666667 1.         0.9        0.96666667] 
#  cross_val_score :  0.9667
# ACC :  [1.         0.96666667 1.         0.9        1.        ] 
#  cross_val_score :  0.9733
# ACC :  [0.96666667 0.96666667 1.         0.93333333 1.        ] 
#  cross_val_score :  0.9733
# MultiOutputClassifier 은 안나온 놈!!!
# ACC :  [0.96666667 0.93333333 1.         0.93333333 1.        ]
#  cross_val_score :  0.9667
# ACC :  [0.93333333 0.9        0.96666667 0.9        0.96666667]
#  cross_val_score :  0.9333
# ACC :  [0.96666667 0.96666667 1.         0.93333333 1.        ] 
#  cross_val_score :  0.9733
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# ACC :  [0.83333333 0.76666667 0.9        0.86666667 0.93333333] 
#  cross_val_score :  0.86
# ACC :  [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] 
#  cross_val_score :  0.78
# ACC :  [1.         0.96666667 1.         0.93333333 1.        ]
#  cross_val_score :  0.98
# ACC :  [0.96666667 0.9        0.96666667 0.93333333 1.        ] 
#  cross_val_score :  0.9533
# ACC :  [0.93333333 0.96666667 1.         0.9        0.96666667]
#  cross_val_score :  0.9533
# ACC :  [0.86666667 0.8        0.93333333 0.7        0.9       ]
#  cross_val_score :  0.84
# ACC :  [0.86666667 0.8        0.93333333 0.7        0.9       ]
#  cross_val_score :  0.84
# ACC :  [0.93333333 0.76666667 0.83333333 0.9        1.        ]
#  cross_val_score :  0.8867
# ACC :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]
#  cross_val_score :  0.9667
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
#=======================================================================================#
