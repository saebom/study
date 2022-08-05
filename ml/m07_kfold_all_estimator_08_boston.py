import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, shuffle=True, random_state=66
# )

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)
#2. 모델 구성, 훈련, 평가, 예측
# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms: 
    try :
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold)
        print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))
    except:
        # continue
        print(name, '은 안나온 놈!!!')
        


#===================================== 결  과 ==========================================#
#ACC :  [0.80125693 0.76317071 0.56809285 0.6400258  0.71991866] 
#  cross_val_score :  0.6985
# ACC :  [0.90118187 0.80633993 0.8021835  0.84156119 0.87348611] 
#  cross_val_score :  0.845
# ACC :  [0.91206034 0.85158944 0.81088253 0.86084496 0.84978706] 
#  cross_val_score :  0.857
# ACC :  [0.79379186 0.81123808 0.57943979 0.62721388 0.70719051] 
#  cross_val_score :  0.7038
# ACC :  [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276]
#  cross_val_score :  0.6471
# ACC :  [0.68382584 0.7482986  0.79447845 0.73222482 0.76021859] 
#  cross_val_score :  0.7438
# ACC :  [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911]
#  cross_val_score :  -0.0135
# ACC :  [0.73383355 0.76745241 0.59979782 0.60616114 0.64658354]
#  cross_val_score :  0.6708
# ACC :  [0.71677604 0.75276545 0.59116613 0.59289916 0.62888608] 
#  cross_val_score :  0.6565
# ACC :  [0.79065879 0.73122203 0.62422925 0.78165167 0.89073576]
#  cross_val_score :  0.7637
# ACC :  [0.93586367 0.84876978 0.77990238 0.88185576 0.92548348] 
#  cross_val_score :  0.8744
# ACC :  [-0.00058757 -0.03146716 -0.00463664 -0.02807276 -0.00298635]
#  cross_val_score :  -0.0136
# ACC :  [-6.07310526 -5.51957093 -6.33482574 -6.36383476 -5.35160828] 
#  cross_val_score :  -5.9286
# ACC :  [0.94613594 0.83639531 0.82713222 0.88506328 0.93115103] 
#  cross_val_score :  0.8852
# ACC :  [0.93235978 0.82415907 0.78740524 0.88879806 0.85766226] 
#  cross_val_score :  0.8581
# ACC :  [0.74957513 0.66191264 0.52848731 0.4033986  0.61481217] 
#  cross_val_score :  0.5916
# ACC :  [nan nan nan nan nan]
#  cross_val_score :  nan
# ACC :  [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856] 
#  cross_val_score :  0.5286
# ACC :  [0.83333255 0.76712443 0.5304997  0.5836223  0.71226555] 
#  cross_val_score :  0.6854
# ACC :  [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384] 
#  cross_val_score :  0.6977
# ACC :  [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854] 
#  cross_val_score :  0.6928
# ACC :  [0.7240751  0.76027388 0.60141929 0.60458689 0.63793473]
#  cross_val_score :  0.6657
# ACC :  [0.71314939 0.79141061 0.60734295 0.61617714 0.66137127] 
#  cross_val_score :  0.6779
# ACC :  [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911]
#  cross_val_score :  -0.0135
# ACC :  [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787] 
#  cross_val_score :  0.6965
# ACC :  [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009] 
#  cross_val_score :  0.713
# ACC :  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 
#  cross_val_score :  0.7128
# ACC :  [0.21361472 0.66767968 0.55829692 0.53411377 0.49113722] 
#  cross_val_score :  0.493
# ACC :  [0.61843443 0.61639175 0.48187755 0.33189363 0.46046524] 
#  cross_val_score :  0.5018
# MultiOutputRegressor 은 안나온 놈!!!
# ACC :  [nan nan nan nan nan]
#  cross_val_score :  nan
# ACC :  [nan nan nan nan nan]
#  cross_val_score :  nan
# ACC :  [nan nan nan nan nan]
#  cross_val_score :  nan
# ACC :  [nan nan nan nan nan]
#  cross_val_score :  nan
# ACC :  [0.2594254  0.33427351 0.263857   0.11914968 0.170599  ] 
#  cross_val_score :  0.2295
# ACC :  [0.58276176 0.565867   0.48689774 0.51545117 0.52049576]
#  cross_val_score :  0.5343
# ACC :  [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377] 
#  cross_val_score :  0.6578
# ACC :  [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868]
#  cross_val_score :  -2.2096
# ACC :  [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313] 
#  cross_val_score :  0.6847
# ACC :  [-6.16495058 -0.22606961 -0.3029979   0.12812791  0.23492048]
#  cross_val_score :  -1.2662
# ACC :  [0.85675147 0.81899657 0.66752992 0.67995723 0.75406999] 
#  cross_val_score :  0.7555
# ACC :  [0.66424055 0.25742657 0.53556064 0.40700866 0.67299593] 
#  cross_val_score :  0.5074
# ACC :  [nan nan nan nan nan] 
#  cross_val_score :  nan
# ACC :  [0.91902267 0.84458273 0.80901126 0.87453205 0.90134144] 
#  cross_val_score :  0.8697
# RegressorChain 은 안나온 놈!!!
# ACC :  [0.80984876 0.80618063 0.58111378 0.63459427 0.72264776]
#  cross_val_score :  0.7109
# ACC :  [0.81125292 0.80010535 0.58888303 0.64008984 0.72362912]
#  cross_val_score :  0.7128
# ACC :  [-2.32160410e+26 -6.21038160e+26 -7.16342273e+26 -9.51368901e+25
#  -6.22386480e+25]
#  cross_val_score :  -3.453832762785863e+26
# ACC :  [0.23475113 0.31583258 0.24121157 0.04946335 0.14020554] 
#  cross_val_score :  0.1963
# StackingRegressor 은 안나온 놈!!!
# ACC :  [0.7775262  0.7266449  0.58485761 0.55649867 0.72196198] 
#  cross_val_score :  0.6735
# ACC :  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 
#  cross_val_score :  0.7128
# ACC :  [0.73773493 0.75660663 0.56519129 0.57692347 0.63088471] 
#  cross_val_score :  0.6535
# VotingRegressor 은 안나온 놈!!!
#=======================================================================================#

