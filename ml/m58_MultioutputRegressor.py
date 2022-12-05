import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

x,y=load_linnerud(return_X_y=True)
# print(x)
# print(y)
'''
[[  5. 162.  60.]
 [  2. 110.  60.]
 [ 12. 101. 101.]
 [ 12. 105.  37.]
 [ 13. 155.  58.]
 [  4. 101.  42.]
 [  8. 101.  38.]
 [  6. 125.  40.]
 [ 15. 200.  40.]
 [ 17. 251. 250.]
 [ 17. 120.  38.]
 [ 13. 210. 115.]
 [ 14. 215. 105.]
 [  1.  50.  50.]
 [  6.  70.  31.]
 [ 12. 210. 120.]
 [  4.  60.  25.]
 [ 11. 230.  80.]
 [ 15. 225.  73.]
 [  2. 110.  43.]]
 ######################################
 [[191.  36.  50.]
 [189.  37.  52.]
 [193.  38.  58.]
 [162.  35.  62.]
 [189.  35.  46.]
 [182.  36.  56.]
 [211.  38.  56.]
 [167.  34.  60.]
 [176.  31.  74.]
 [154.  33.  56.]
 [169.  34.  50.]
 [166.  33.  52.]
 [154.  34.  64.]
 [247.  46.  50.]
 [193.  36.  46.]
 [202.  37.  62.]
 [176.  37.  54.]
 [157.  32.  52.]
 [156.  33.  54.]
 [138.  33.  68.]]
 
'''

# print(x.shape,y.shape) #(20, 3) (20, 3) 

model=Ridge()
model.fit(x,y)
print(model.predict([[2,110,43]])) #[[187.32842123  37.0873515   55.40215097]]

model=XGBRegressor()
model.fit(x,y)
print(model.predict([[2,110,43]])) #[[138.00215   33.001656  67.99831 ]]

# model=CatBoostRegressor()
# model.fit(x,y)
# print(model.predict([[2,110,43]]))
# Currently only multi-regression, multilabel and survival objectives work with multidimensional target => multioutputregressor로 

model=MultiOutputRegressor(CatBoostRegressor())
model.fit(x,y)
print(model.predict([[2,110,43]])) #[[138.97756017  33.09066774  67.61547996]]

# model=LGBMRegressor()
# model.fit(x,y)
# print(model.predict([[2,110,43]]))
# ValueError: y should be a 1d array, got an array of shape (20, 3) instead => multioutputregressor로

model=MultiOutputRegressor(LGBMRegressor())
model.fit(x,y)
print(model.predict([[2,110,43]])) #[[178.6  35.4  56.1]]