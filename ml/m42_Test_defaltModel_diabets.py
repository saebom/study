import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

# x = np.delete(x, [0, 1, 5], axis=1)
# print(x.shape)  # (442, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=72, train_size=0.8
)

#2. 모델
model = LinearRegression()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)    
print('r2 : ', result)

y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", acc)


#=================================== 결과 =====================================#
# r2 :  0.6579209558684551
#==============================================================================#