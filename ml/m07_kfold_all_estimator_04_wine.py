import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_wine()
x, y = datasets.data, datasets.target

print(x.shape)  # (178, 13)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=66
# )

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=72)

#2. 모델 구성, 훈련, 평가, 예측
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms: 
    try :
        model = algorithm()
        scores = cross_val_score(model, x, y, cv = kfold)
        print('ACC : ', scores, '\n cross_val_score', round(np.mean(scores), 4))
    except:
        # continue
        print(name, '은 안나온 놈!!!')
        

#===================================== 결  과 ==========================================#
# ACC :  [0.97222222 0.97222222 0.94444444 0.97142857 0.88571429] 
#  cross_val_score 0.9492
# ACC :  [0.91666667 0.97222222 0.88888889 0.97142857 0.97142857] 
#  cross_val_score 0.9441
# ACC :  [0.22222222 0.5        0.47222222 0.37142857 0.42857143]
#  cross_val_score 0.3989
# ACC :  [0.83333333 0.94444444 0.97222222 0.91428571 0.94285714] 
#  cross_val_score 0.9214
# ACC :  [       nan        nan 0.91666667        nan        nan] 
#  cross_val_score nan
# ClassifierChain 은 안나온 놈!!!
# ACC :  [0.55555556 0.63888889 0.72222222 0.68571429 0.74285714]
#  cross_val_score 0.669
# ACC :  [0.80555556 0.80555556 0.86111111 0.94285714 0.85714286] 
#  cross_val_score 0.8544
# ACC :  [0.22222222 0.5        0.47222222 0.37142857 0.42857143]
#  cross_val_score 0.3989
# ACC :  [0.91666667 0.94444444 0.72222222 0.88571429 0.88571429] 
#  cross_val_score 0.871
# ACC :  [0.94444444 1.         0.97222222 0.97142857 1.        ] 
#  cross_val_score 0.9776
# ACC :  [0.97222222 0.97222222 0.97222222 0.94285714 1.        ]
#  cross_val_score 0.9719
# ACC :  [0.47222222 0.38888889 0.55555556 0.42857143 0.48571429] 
#  cross_val_score 0.4662
# ACC :  [0.86111111 0.91666667 0.86111111 0.97142857 0.91428571] 
#  cross_val_score 0.9049
# ACC :  [0.94444444 0.94444444 0.91666667 0.97142857 1.        ] 
#  cross_val_score 0.9554
# ACC :  [0.63888889 0.63888889 0.72222222 0.77142857 0.8       ] 
#  cross_val_score 0.7143
# ACC :  [0.58333333 0.27777778 0.52777778 0.54285714 0.48571429] 
#  cross_val_score 0.4835
# ACC :  [0.58333333 0.27777778 0.52777778 0.54285714 0.48571429] 
#  cross_val_score 0.4835
# ACC :  [1.         1.         0.97222222 1.         0.97142857] 
#  cross_val_score 0.9887
# ACC :  [0.86111111 0.80555556 0.41666667 0.82857143 0.65714286] 
#  cross_val_score 0.7138
# ACC :  [0.91666667 1.         0.94444444 0.94285714 0.94285714] 
#  cross_val_score 0.9494
# ACC :  [0.91666667 0.97222222 0.97222222 0.94285714 0.94285714] 
#  cross_val_score 0.9494
# ACC :  [0.86111111 0.5        0.27777778 0.94285714 0.08571429] 
#  cross_val_score 0.5335
# MultiOutputClassifier 은 안나온 놈!!!
# ACC :  [0.83333333 0.77777778 0.83333333 0.94285714 0.88571429]
#  cross_val_score 0.8546
# ACC :  [0.63888889 0.69444444 0.75       0.82857143 0.71428571]
#  cross_val_score 0.7252
# ACC :  [0.80555556 0.75       0.91666667 0.91428571 0.85714286] 
#  cross_val_score 0.8487
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# ACC :  [0.55555556 0.69444444 0.69444444 0.62857143 0.62857143] 
#  cross_val_score 0.6403
# ACC :  [0.61111111 0.63888889 0.63888889 0.45714286 0.65714286] 
#  cross_val_score 0.6006
# ACC :  [1.         1.         0.97222222 1.         1.        ]
#  cross_val_score 0.9944
# ACC :  [nan nan nan nan nan] 
#  cross_val_score nan
# ACC :  [0.97222222 1.         0.97222222 0.97142857 1.        ] 
#  cross_val_score 0.9832
# ACC :  [1.         1.         0.97222222 1.         1.        ]
#  cross_val_score 0.9944
# ACC :  [1.         1.         0.97222222 1.         1.        ] 
#  cross_val_score 0.9944
# ACC :  [0.47222222 0.69444444 0.66666667 0.54285714 0.68571429] 
#  cross_val_score 0.6124
# ACC :  [0.47222222 0.69444444 0.72222222 0.68571429 0.68571429]
#  cross_val_score 0.6521
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
#=======================================================================================#


