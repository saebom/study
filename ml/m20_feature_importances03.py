##################################################################
# 실습
##################################################################


import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(datasets.DESCR)
print(x.shape) # (442, 10)
# print(datasets.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets['feature_names'])

# drop_features
# x = np.delete(x, [0, 1], axis=1)
x = np.delete(x, [0, 1, 7], axis=1)
print(x.shape)  # (442, 7)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72  
)


#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score : ', result)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test,)
acc = r2_score(y_test, y_predict)
print('r2_score : ', acc)

print('==============================')
print(model, ': ', model.feature_importances_)


#======================================================== age, sex, s4 삭제  결과 ========================================================#
# r2_score :  0.17272382158077348
# r2_score :  0.5019665005381797
# r2_score :  0.516525511785247
# r2_score :  0.39577733354774336
# ====================================
# DecisionTreeRegressor()     :  [0.2717281  0.05571535 0.05573649 0.08513617 0.08822724 0.36978434  0.07367231]
# RandomForestRegressor()     :  [0.26373871 0.09194354 0.06630295 0.07656909 0.07532669 0.34059179  0.08552725]
# GradientBoostingRegressor() :  [0.27250506 0.07182163 0.03339135 0.08749606 0.05195973 0.41642431  0.06640185]
# XGBRegressor()              :  [0.14754291 0.06889056 0.05587956 0.10462266 0.09215297 0.43945208  0.09145921]

#========================================================== age, sex 삭제  결과 ==========================================================#
# r2_score :  0.10728926879031697
# r2_score :  0.5093135351756595
# r2_score :  0.5044895475145719
# r2_score :  0.4048820536919818
# ====================================
# DecisionTreeRegressor()     :  [0.26874255 0.05029475 0.05518293 0.093089   0.08084259 0.00751664 0.3704983  0.07383324]
# RandomForestRegressor()     :  [0.28608518 0.08393274 0.05912148 0.06969278 0.07520337 0.02312073 0.31994145 0.08290228]
# GradientBoostingRegressor() :  [0.26858984 0.06878119 0.03553935 0.0828121  0.05580147 0.00925717 0.41209537 0.06712351]
# XGBRegressor()              :  [0.14416957 0.07217783 0.04523559 0.07164037 0.08108158 0.08678748 0.4092454  0.08966216]
#================================================================================================================================#


############################################################  컬럼 삭제 전  #####################3##########################################
# r2_score :  0.0931371643680513
# r2_score :  0.5508741236890558
# r2_score :  0.5236608835304826
# r2_score :  0.5149645205811251
# ====================================
# DecisionTreeRegressor()     :  [0.02266884 0.00732187 0.27144689 0.05082537 0.04571841 0.08733958 0.08638132 0.00380016 0.35783385 0.06666372]
# RandomForestRegressor()     :  [0.05796704 0.00990173 0.27267557 0.07638843 0.04804558 0.06195474 0.06143343 0.02265756 0.31357968 0.07539626]
# GradientBoostingRegressor() :  [0.04234091 0.00748993 0.27156352 0.07662691 0.02655896 0.05376873 0.05729347 0.01213045 0.39711355 0.05511357]
# XGBRegressor()              :  [0.02859626 0.03671591 0.18356994 0.06468294 0.04937157 0.0577106  0.07647511 0.08029433 0.37062448 0.05195891]
