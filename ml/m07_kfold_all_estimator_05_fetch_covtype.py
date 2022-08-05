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
# ACC :  [0.55381531 0.41932652 0.53042977 0.56238275 0.52622158]
#  cross_val_score ;  0.5184
# ACC :  [0.96207499 0.96118861 0.96199721 0.96131736 0.9615239 ]
#  cross_val_score ;  0.9616
# ACC :  [0.63382185 0.63033657 0.6291544  0.6329237  0.63001497]
#  cross_val_score ;  0.6313
# ACC :  [0.6722718  0.68255553 0.67342214 0.68373178 0.69576255] 
#  cross_val_score ;  0.6815
# ACC :  [nan nan nan nan nan]   
#  cross_val_score ;  nan        
# ClassifierChain 은 안나온 놈!!!
# ACC :  [nan nan nan nan nan] 
#  cross_val_score ;  nan      
# ACC :  [0.93867628 0.93948521 0.93837455 0.94015594 0.94011291] 
#  cross_val_score ;  0.9394
# ACC :  [0.48858463 0.48664837 0.48889004 0.48765942 0.48621366] 
#  cross_val_score ;  0.4876
# ACC :  [0.857835   0.86207757 0.86173216 0.87752362 0.84928831] 
#  cross_val_score ;  0.8617
# ACC :  [0.9542955  0.95308211 0.95264281 0.95442419 0.9524793 ] 
#  cross_val_score ;  0.9534
# ACC :  [0.4565889  0.45806046 0.45982857 0.46019862 0.46023304] 
#  cross_val_score ;  0.459
# ACC :  [nan nan nan nan nan] 
#  cross_val_score ;  nan   
#=======================================================================================#

