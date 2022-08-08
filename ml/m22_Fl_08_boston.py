import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

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



#결과비교

#======================================  [] 삭제  결과 =======================================#
# 1. DecisionTree
# 기존 r2 : 
# 컬럼 삭제 후 r2 :  

# 2. RandomForestClassifier
# 기존 r2 : 
# 컬럼 삭제 후 r2 : 

# 3. GradientBoostingClassifier
# 기존 r2 :  
# 컬럼 삭제 후 r2 : 

# 4. XGBClassifier
# 기존 r2 : 
# 컬럼 삭제 후 r2 : 
#=========================================================================================================================#
