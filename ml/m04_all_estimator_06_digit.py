import numpy as np
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성, 훈련, 평가, 예측
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms: 
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        # continue
        print(name, '은 안나온 놈!!!')
        
        

#===================================== 결  과 ==========================================#
# AdaBoostClassifier 의 정답률 :  0.26296296296296295
# BaggingClassifier 의 정답률 :  0.95
# BernoulliNB 의 정답률 :  0.8722222222222222
# CalibratedClassifierCV 의 정답률 :  0.9685185185185186
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 은 안나온 놈!!!
# DecisionTreeClassifier 의 정답률 :  0.8611111111111112
# DummyClassifier 의 정답률 :  0.08888888888888889
# ExtraTreeClassifier 의 정답률 :  0.7444444444444445
# ExtraTreesClassifier 의 정답률 :  0.9796296296296296
# GaussianNB 의 정답률 :  0.8222222222222222
# GaussianProcessClassifier 의 정답률 :  0.912962962962963
# GradientBoostingClassifier 의 정답률 :  0.9740740740740741
# HistGradientBoostingClassifier 의 정답률 :  0.9796296296296296
# KNeighborsClassifier 의 정답률 :  0.9185185185185185
# LabelPropagation 의 정답률 :  0.9018518518518519
# LabelSpreading 의 정답률 :  0.9018518518518519
# LinearDiscriminantAnalysis 의 정답률 :  0.9481481481481482
# LinearSVC 의 정답률 :  0.9648148148148148
# LogisticRegression 의 정답률 :  0.9722222222222222
# LogisticRegressionCV 의 정답률 :  0.9611111111111111
# MLPClassifier 의 정답률 :  0.9777777777777777
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 은 안나온 놈!!!
# NearestCentroid 의 정답률 :  0.7074074074074074
# NuSVC 의 정답률 :  0.9222222222222223
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 :  0.9481481481481482
# Perceptron 의 정답률 :  0.9351851851851852
# QuadraticDiscriminantAnalysis 의 정답률 :  0.8555555555555555
# RadiusNeighborsClassifier 은 안나온 놈!!!
# RandomForestClassifier 의 정답률 :  0.9814814814814815
# RidgeClassifier 의 정답률 :  0.9351851851851852
# RidgeClassifierCV 의 정답률 :  0.937037037037037
# SGDClassifier 의 정답률 :  0.9462962962962963
# SVC 의 정답률 :  0.9648148148148148
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
#=======================================================================================#

