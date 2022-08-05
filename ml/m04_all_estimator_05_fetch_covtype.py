import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
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
# AdaBoostClassifier 의 정답률 :  0.3941332384798972
# BaggingClassifier 의 정답률 :  0.9584977969524509
# BernoulliNB 의 정답률 :  0.6334392785019277
# CalibratedClassifierCV 의 정답률 :  0.7132997521571507
# CategoricalNB 의 정답률 :  0.6338752983293556
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 정답률 :  0.6204734257389388
# DecisionTreeClassifier 의 정답률 :  0.9335184505232238
# DummyClassifier 의 정답률 :  0.48838236644024235
# ExtraTreeClassifier 의 정답률 :  0.8390054158252249
# ExtraTreesClassifier 의 정답률 :  0.9512002019460253
# GaussianNB 의 정답률 :  0.08996351202496787
# GaussianProcessClassifier 은 안나온 놈!!!
# GradientBoostingClassifier 의 정답률 :  0.771301863411052
# HistGradientBoostingClassifier 의 정답률 :  0.780727923627685
# KNeighborsClassifier 의 정답률 :  0.9331398017257205
# LabelPropagation 은 안나온 놈!!!
# LabelSpreading 은 안나온 놈!!!
# LinearDiscriminantAnalysis 의 정답률 :  0.6797376996511841
# LinearSVC 의 정답률 :  0.7129956856985497
# LogisticRegression 의 정답률 :  0.7197712043326602
# LogisticRegressionCV 의 정답률 :  0.7246305305672848
# MLPClassifier 의 정답률 :  0.8353107214980723
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 정답률 :  0.643731641270424
# NearestCentroid 의 정답률 :  0.38909606205250596
# NuSVC 은 안나온 놈!!!
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 :  0.6284250504865063
# Perceptron 의 정답률 :  0.5945818799339085
# QuadraticDiscriminantAnalysis 의 정답률 :  0.0847656967137874
# RadiusNeighborsClassifier 은 안나온 놈!!!
# RandomForestClassifier 의 정답률 :  0.9531967137874059
# RidgeClassifier 의 정답률 :  0.7012977326968973
# RidgeClassifierCV 의 정답률 :  0.7013321553148522
# SGDClassifier 의 정답률 :  0.7121293831466863
# SVC 의 정답률 :  0.7706019368459702
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
#=======================================================================================#

