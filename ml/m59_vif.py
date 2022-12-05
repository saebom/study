# 다중공선성 = 전체변수 사이의 상관관계

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

dataset = fetch_california_housing()
# print(dataset.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(20640, 8) (20640,)   
# print(type(x)) #<class 'numpy.ndarray'>

x=pd.DataFrame(x, columns=dataset.feature_names)
# print(x)

# 다중공선성 확인
# vif = pd.DataFrame()
# vif['vif Factor']=[variance_inflation_factor(
#     x.values,i) for i in range(x.shape[1])]
# vif['features'] = x.columns
# print(vif)
'''
5~10 이상이면 높다고 판단 (통상적으로 5이상이면 높다고 판단, 피쳐가 너무 많을경우 10)
   vif Factor    features
0   11.511140      MedInc
1    7.195917    HouseAge
2   45.993601    AveRooms
3   43.590314   AveBedrms
4    2.935745  Population
5    1.095243    AveOccup
6  559.874071    Latitude
7  633.711654   Longitude
높은 순으로 제거해보자
'''

# 피쳐 제거
drop_feature = ['AveRooms']
x = x.drop(drop_feature, axis=1)

# 제거 후 확인
vif = pd.DataFrame()
vif['vif Factor']=[variance_inflation_factor(
    x.values,i) for i in range(x.shape[1])]
vif['features'] = x.columns
print(vif)

'''
'Longitude','AveBedrms','Latitude' 제거
   vif Factor    features
0    5.111195      MedInc
1    3.276300    HouseAge
2    5.137411    AveRooms
3    2.136316  Population
4    1.094552    AveOccup
'''

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=123,train_size=0.8)
model=RandomForestRegressor()
model.fit(x_train,y_train)
result=model.score(x_test,y_test)
print('model.score:',result) 

# model.score: 0.7349000768550964 <- 'Longitude' 삭제
# model.score: 0.7456966311095827 <- 'Latitude' 삭제
# model.score: 0.8175927585219248 <- 'AveRooms' 삭제
# model.score: 0.6710152383250181 <- 'Longitude','AveBedrms','Latitude' 삭제 