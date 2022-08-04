import numpy as ny
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=66
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression     #LogisticRegression은 분류모델, LinearRegression 회귀모델
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = LinearSVR()
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()


#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
results = model.score(x_test, y_test)                                         
print('결과 r2 : ', results)


#===================================== 결  과 ==========================================#
# LinearSVR() 결과 r2 :  0.7524301881386657
# LinearRegression() 결과 r2 :  0.8044888426543624
# KNeighborsRegressor() 결과 r2 :  0.8528861673934228
# DecisionTreeRegressor() 결과 결과 r2 :  0.6510328077668611
# RandomForestRegressor() 결과 r2 : 0.8920124870646153
#=======================================================================================#

