import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target


# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=72
# )
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (309, 10) (133, 10) (309,) (133,)
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=72)

#2. 모델 구성
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
# ACC :  [0.65305886 0.47586081 0.44215345 0.3444933  0.41018908] 
#  cross_val_score :  0.4652
# ACC :  [0.5750322  0.40962986 0.32462492 0.28462794 0.37115037] 
#  cross_val_score :  0.393
# ACC :  [0.4720695  0.39186192 0.2854109  0.3008096  0.32632837] 
#  cross_val_score :  0.3553
# ACC :  [0.65307472 0.45623161 0.45581838 0.34507203 0.4222871 ]
#  cross_val_score :  0.4665
# ACC :  [0.66538954 0.36547845 0.36913219 0.33211899 0.41982916] 
#  cross_val_score :  0.4304
# ACC :  [ 0.03031115 -0.10996466 -0.40640569 -0.19265054 -0.09741987]
#  cross_val_score :  -0.1552
# ACC :  [-0.01440045 -0.00144529 -0.0122963  -0.1243925  -0.02034141] 
#  cross_val_score :  -0.0346
# ACC :  [-0.00532826  0.00837779 -0.00385606 -0.1149083  -0.01268163]
#  cross_val_score :  -0.0257
# ACC :  [0.54162015 0.43244402 0.42360469 0.34873279 0.34476681] 
#  cross_val_score :  0.4182
# ACC :  [-0.08122282 -0.19091792  0.09894974 -0.20473367 -0.31494284] 
#  cross_val_score :  -0.1386
# ACC :  [0.55334172 0.40873563 0.34218935 0.37698383 0.36367588] 
#  cross_val_score :  0.409
# ACC :  [-0.00694777  0.00495142 -0.00591597 -0.09562745 -0.0146168 ]
#  cross_val_score :  -0.0236
# ACC :  [-19.91521535 -15.46302828 -11.60735408  -8.59380589  -6.15991498] 
#  cross_val_score :  -12.3479
# ACC :  [0.52602278 0.45021625 0.35802268 0.33050582 0.32721871] 
#  cross_val_score :  0.3984
# ACC :  [0.53485455 0.36650641 0.32375449 0.36614474 0.3161729 ] 
#  cross_val_score :  0.3815
# ACC :  [0.65695829 0.45608565 0.45655918 0.32347846 0.42694311] 
#  cross_val_score :  0.464
# ACC :  [nan nan nan nan nan]
#  cross_val_score :  nan
# ACC :  [0.54033516 0.3051046  0.37200469 0.3615798  0.21857686] 
#  cross_val_score :  0.3595
# ACC :  [-3.4081568  -4.2278114  -3.69459382 -2.89084926 -4.10828901] 
#  cross_val_score :  -3.6659
# ACC :  [ 0.65792096  0.13824263  0.41139904 -2.9498601   0.4257726 ] 
#  cross_val_score :  -0.2633
# ACC :  [0.65666451 0.37235963 0.45152833 0.37649658 0.40971684] 
#  cross_val_score :  0.4534
# ACC :  [0.37852179 0.40250089 0.29793757 0.24562294 0.30690083]
#  cross_val_score :  0.3263
# ACC :  [0.6518881  0.46029612 0.45020633 0.34746135 0.40780121] 
#  cross_val_score :  0.4635
# ACC :  [0.4154746  0.42364998 0.31321341 0.26705601 0.32273016]
#  cross_val_score :  0.3484
# ACC :  [0.65102723 0.46069739 0.45002017 0.34849229 0.40726012] 
#  cross_val_score :  0.4635
# ACC :  [0.64596124 0.47933712 0.45124827 0.35334386 0.4058545 ] 
#  cross_val_score :  0.4671
# ACC :  [0.65792096 0.4626711  0.45682648 0.33358517 0.4257726 ] 
#  cross_val_score :  0.4674
# ACC :  [-0.45655053 -0.40892312 -0.49776011 -0.05616653 -0.58096025]
#  cross_val_score :  -0.4001
# ACC :  [-3.08509007 -3.19993611 -3.18280281 -2.30196674 -3.62612331] 
#  cross_val_score :  -3.0792
# MultiOutputRegressor 은 안나온 놈!!!
# ACC :  [nan nan nan nan nan]
#  cross_val_score :  nan
# ACC :  [nan nan nan nan nan]
#  cross_val_score :  nan
# ACC :  [nan nan nan nan nan]
#  cross_val_score :  nan
# ACC :  [nan nan nan nan nan]
#  cross_val_score :  nan
# ACC :  [0.15030607 0.18146833 0.13322945 0.05737185 0.12219839] 
#  cross_val_score :  0.1289
# ACC :  [0.41570596 0.42311625 0.20837989 0.22818337 0.32033198]
#  cross_val_score :  0.3191
# ACC :  [0.65382154 0.4867191  0.41344735 0.32229173 0.41045784] 
#  cross_val_score :  0.4573
# ACC :  [-0.8475176  -2.25961928 -0.75922057 -1.09499232 -1.4271007 ]
#  cross_val_score :  -1.2777
# ACC :  [0.6545418  0.4363136  0.45789772 0.34937192 0.43364925] 
#  cross_val_score :  0.4664
# ACC :  [0.58384085 0.41294729 0.45606012 0.40758086 0.32272268] 
#  cross_val_score :  0.4366
# ACC :  [0.38811306 0.34417486 0.32350509 0.24398703 0.25259878] 
#  cross_val_score :  0.3105
# ACC :  [0.13432858 0.28169124 0.13667338 0.25388121 0.11240772] 
#  cross_val_score :  0.1838
# ACC :  [-0.01440045 -0.00144529 -0.0122963  -0.1243925  -0.02034141] 
#  cross_val_score :  -0.0346
# ACC :  [0.56148426 0.46008379 0.32556656 0.33895833 0.38946403] 
#  cross_val_score :  0.4151
# RegressorChain 은 안나온 놈!!!
# ACC :  [0.50591093 0.41968767 0.40690793 0.33472454 0.32744944]
#  cross_val_score :  0.3989
# ACC :  [0.64326203 0.45597794 0.46183132 0.35609644 0.41572376] 
#  cross_val_score :  0.4666
# ACC :  [0.48461418 0.40524005 0.39671956 0.34384918 0.30002554] 
#  cross_val_score :  0.3861
# ACC :  [0.14455787 0.1753805  0.11205524 0.14291688 0.09415005] 
#  cross_val_score :  0.1338
# StackingRegressor 은 안나온 놈!!!
# ACC :  [0.65732776 0.4247423  0.45104261 0.33966531 0.4190682 ] 
#  cross_val_score :  0.4584
# ACC :  [0.65792096 0.4626711  0.45682648 0.33358517 0.4257726 ]
#  cross_val_score :  0.4674
# ACC :  [-0.00749754  0.00581963 -0.00614303 -0.11708604 -0.01498616]
#  cross_val_score :  -0.028
# VotingRegressor 은 안나온 놈!!!
#=======================================================================================#


