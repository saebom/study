import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(datasets.DESCR)
print(x.shape)  # (1797, 64)

# drop_features
x = np.delete(x, [0, 2, 7, 8, 11, 14, 15, 16, 17, 22, 23, 25, 
                  31, 32, 33, 35, 38, 39, 40, 46, 47, 48, 55, 56, 57, 63], axis=1)
print(x.shape)  # (1797, 38)


from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72
    )
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)   # train은 fit_transform, test는 transform으로 overfit(과적합)이 안 잡힘



#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()



#3. 훈련
model.fit(x_train, y_train) # make_pipeline에서의 fit은 fit_transform이 적용됨


#4. 평가, 예측
result = model.score(x_test, y_test)    # make_pipeline에서의 model.score는 transform이 적용됨
# print('model.score : ', result) 

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test,)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

print('==============================')
print(model, ': ', model.feature_importances_)


#결과비교

#======================================  [] 삭제  결과 =======================================#
# 1. DecisionTree
# 기존 acc : 0.8888888888888888
# 컬럼 삭제 후 acc :  0.8916666666666667 

# 2. RandomForestClassifier
# 기존 acc : 0.9777777777777777
# 컬럼 삭제 후 acc :  0.9833333333333333

# 3. GradientBoostingClassifier
# 기존 acc :  0.9805555555555555
# 컬럼 삭제 후 acc :  0.9861111111111112

# 4. XGBClassifier
# 기존 acc : 0.975
# 컬럼 삭제 후 acc : 0.9861111111111112
#=========================================================================================================================#
