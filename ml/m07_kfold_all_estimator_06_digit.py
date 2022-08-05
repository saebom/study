import numpy as np
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=72
# )

n_splits = 5
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
# ACC :  [0.25833333 0.225      0.24791086 0.27019499 0.24791086] 
#  cross_val_score ;  0.2499
# ACC :  [0.96666667 0.91111111 0.93036212 0.92200557 0.93593315] 
#  cross_val_score ;  0.9332
# ACC :  [0.86388889 0.86666667 0.85236769 0.85236769 0.83844011] 
#  cross_val_score ;  0.8547
# ACC :  [0.95555556 0.95       0.95821727 0.97493036 0.96935933] 
#  cross_val_score ;  0.9616
# ACC :  [nan nan nan nan nan] 
#  cross_val_score ;  nan
# ClassifierChain 은 안나온 놈!!!
# ACC :  [0.81944444 0.83055556 0.76601671 0.81337047 0.84958217] 
#  cross_val_score ;  0.8158
# ACC :  [0.88888889 0.84444444 0.82172702 0.86350975 0.86908078] 
#  cross_val_score ;  0.8575
# ACC :  [0.08333333 0.06944444 0.06685237 0.09749304 0.08356546]
#  cross_val_score ;  0.0801
# ACC :  [0.80555556 0.78611111 0.76601671 0.77994429 0.79387187] 
#  cross_val_score ;  0.7863
# ACC :  [0.98611111 0.975      0.97214485 0.98885794 0.98607242] 
#  cross_val_score ;  0.9816
# ACC :  [0.82222222 0.83333333 0.83286908 0.84401114 0.84122563] 
#  cross_val_score ;  0.8347
# ACC :  [0.11666667 0.08611111 0.1086351  0.11699164 0.08356546] 
#  cross_val_score ;  0.1024
# ACC :  [0.98055556 0.95       0.96100279 0.96100279 0.96935933] 
#  cross_val_score ;  0.9644
# ACC :  [0.98611111 0.95277778 0.96935933 0.96935933 0.97214485] 
#  cross_val_score ;  0.97
# ACC :  [0.99166667 0.97777778 0.97771588 0.99442897 0.98885794] 
#  cross_val_score ;  0.9861
# ACC :  [0.11111111 0.10555556 0.08913649 0.09192201 0.09749304] 
#  cross_val_score ;  0.099
# ACC :  [0.11111111 0.10555556 0.08913649 0.09192201 0.09749304] 
#  cross_val_score ;  0.099
# ACC :  [0.94444444 0.95555556 0.93593315 0.96935933 0.95543175] 
#  cross_val_score ;  0.9521
# ACC :  [0.93333333 0.94722222 0.93593315 0.95543175 0.95821727] 
#  cross_val_score ;  0.946
# ACC :  [0.96388889 0.95       0.9637883  0.96657382 0.97214485] 
#  cross_val_score ;  0.9633
# ACC :  [0.96944444 0.95833333 0.95543175 0.97214485 0.98328691] 
#  cross_val_score ;  0.9677
# ACC :  [0.97222222 0.98611111 0.94707521 0.98050139 0.97493036] 
#  cross_val_score ;  0.9722
# MultiOutputClassifier 은 안나온 놈!!!
# ACC :  [0.90277778 0.88333333 0.8913649  0.91922006 0.90250696]
#  cross_val_score ;  0.8998
# ACC :  [0.90555556 0.9        0.88300836 0.9275766  0.89693593] 
#  cross_val_score ;  0.9026
# ACC :  [0.97222222 0.95277778 0.96100279 0.97771588 0.96100279] 
#  cross_val_score ;  0.9649
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# ACC :  [0.95277778 0.95833333 0.92200557 0.96935933 0.94986072] 
#  cross_val_score ;  0.9505
# ACC :  [0.96111111 0.95       0.93871866 0.95821727 0.95543175] 
#  cross_val_score ;  0.9527
# ACC :  [0.79722222 0.85       0.79665738 0.83844011 0.91086351] 
#  cross_val_score ;  0.8386
# ACC :  [nan nan nan nan nan] 
#  cross_val_score ;  nan
# ACC :  [0.98055556 0.97222222 0.96935933 0.98050139 0.97214485] 
#  cross_val_score ;  0.975
# ACC :  [0.93611111 0.95       0.89972145 0.94428969 0.94707521] 
#  cross_val_score ;  0.9354
# ACC :  [0.93611111 0.95       0.89972145 0.94428969 0.94707521] 
#  cross_val_score ;  0.9354
# ACC :  [0.96944444 0.94722222 0.94428969 0.9637883  0.93871866] 
#  cross_val_score ;  0.9527
# ACC :  [0.98888889 0.98611111 0.98607242 0.99442897 0.98328691] 
#  cross_val_score ;  0.9878
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
#=======================================================================================#

