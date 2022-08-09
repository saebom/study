import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(datasets.feature_names)    # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']


#pandas의 columns를 통해 컬럼명 호출
x = pd.DataFrame(x, columns=[['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']])

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72
    )


#2. 모델구성
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

# model1 = DecisionTreeRegressor()
# model2 = RandomForestRegressor()
# model3 = GradientBoostingRegressor()
model4 = XGBRegressor()


#3. 훈련
# model1.fit(x_train, y_train) 
# model2.fit(x_train, y_train) 
# model3.fit(x_train, y_train) 
model4.fit(x_train, y_train) 


#4. 평가, 예측
# result = model.score(x_test, y_test)
# print('model.score : ', result)

# from sklearn.metrics import accuracy_score
# y_predict = model.predict(x_test,)
# acc = accuracy_score(y_test, y_predict)
# print('accuracy_score : ', acc)

# print('==============================')
# print(model1, ': ', model1.feature_importances_)
# print(model2, ': ', model2.feature_importances_)
# print(model3, ': ', model3.feature_importances_)
# print(model4, ': ', model4.feature_importances_)


# 그래프
import matplotlib.pyplot as plt

# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)
   
    
# plot_feature_importances(model)
# plt.show()

from xgboost.plotting import plot_importance
plot_importance(model4)
plt.show()

