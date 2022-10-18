import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
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

model1 = DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = GradientBoostingRegressor()
model4 = XGBRegressor()



#3. 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)


#4. 평가, 예측
# result = model.score(x_test, y_test)
# print('model.score : ', result)

# from sklearn.metrics import accuracy_score
# y_predict = model.predict(x_test,)
# acc = accuracy_score(y_test, y_predict)
# print('accuracy_score : ', acc)

print('==============================')
print(model1, ': ', model1.feature_importances_)
print(model2, ': ', model2.feature_importances_)
print(model3, ': ', model3.feature_importances_)
print(model4, ': ', model4.feature_importances_)


# 그래프
import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
   
    
plt.figure(figsize=(20, 10))
plt.suptitle('Feature_importances',fontsize=20)
plt.subplot(2, 2, 1)
plot_feature_importances(model1)
plt.title(model1) 

plt.subplot(2, 2, 2)
plot_feature_importances(model2)
plt.title(model2) 

plt.subplot(2, 2, 3)
plot_feature_importances(model3)
plt.title(model3) 
    
plt.subplot(2, 2, 4)
plot_feature_importances(model4)
plt.title("XGBooster()") 
    
# plot_feature_importances(model)
plt.show()

#============================================   결과 ========================================================#
# r2_score :  0.0931371643680513
# r2_score :  0.5508741236890558
# r2_score :  0.5236608835304826
# r2_score :  0.5149645205811251
# ==============================
# DecisionTreeRegressor()     :  [0.02266884 0.00732187 0.27144689 0.05082537 0.04571841 0.08733958 0.08638132 0.00380016 0.35783385 0.06666372]
# RandomForestRegressor()     :  [0.05796704 0.00990173 0.27267557 0.07638843 0.04804558 0.06195474 0.06143343 0.02265756 0.31357968 0.07539626]
# GradientBoostingRegressor() :  [0.04234091 0.00748993 0.27156352 0.07662691 0.02655896 0.05376873 0.05729347 0.01213045 0.39711355 0.05511357]
# XGBRegressor()              :  [0.02859626 0.03671591 0.18356994 0.06468294 0.04937157 0.0577106 0.07647511 0.08029433 0.37062448 0.05195891]
