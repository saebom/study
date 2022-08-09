import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# drop_features
x = np.delete(x, [1, 3], axis=1)
print(x.shape)  # (506, 11)

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
# print('model.score : ', result)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test,)
acc = r2_score(y_test, y_predict)
print('r2_score : ', acc)

print('==============================')
print(model, ': ', model.feature_importances_)



#결과비교

#======================================  [] 삭제  결과 =======================================#
# 1. DecisionTree
# 기존 r2 : 0.6539277039531357
# 컬럼 삭제 후 r2 :  0.7098399245286287
# [0.04683688 0.00262381 0.00783617 0.00239571 0.03087515 0.622466
#  0.00626991 0.05253834 0.00067791 0.00651704 0.01119528 0.01894052
#  0.19082729]

# 2. RandomForestClassifier
# 기존 r2 : 0.8081132751218749
# 컬럼 삭제 후 r2 : 0.8053638367075079
# [0.03165664 0.00086964 0.00690294 0.0010048  0.01826227 0.45287136
#  0.01200775 0.05064467 0.00339512 0.01526872 0.01127591 0.00872535
#  0.38711482]

# 3. GradientBoostingClassifier
# 기존 r2 :  0.8200322919242531
# 컬럼 삭제 후 r2 :  0.8348950537832021
# [0.03379907 0.00057925 0.00147123 0.00112464 0.01955891 0.43337441
#  0.00882488 0.05765186 0.00248089 0.01484687 0.02426995 0.00662302
#  0.39539501]

# 4. XGBClassifier
# 기존 r2 : 0.8099618738424226
# 컬럼 삭제 후 r2 :  0.8071753899464104
# [0.01682705 0.00902306 0.00725174 0.0005245  0.04748791 0.38118756
#  0.00760658 0.05244936 0.02684627 0.03905421 0.03586074 0.00881991
#  0.36706114]
#=========================================================================================================================#
