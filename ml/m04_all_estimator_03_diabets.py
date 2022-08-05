
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (309, 10) (133, 10) (309,) (133,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms: 
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 정답률 : ', r2)
    except:
        # continue
        print(name, '은 안나온 놈!!!')
        


#===================================== 결  과 ==========================================#
# ARDRegression 의 정답률 :  0.6261983086953462
# AdaBoostRegressor 의 정답률 :  0.5494249345034663
# BaggingRegressor 의 정답률 :  0.533266417088649
# BayesianRidge 의 정답률 :  0.623314043118258
# CCA 의 정답률 :  0.635317998524362
# DecisionTreeRegressor 의 정답률 :  0.038037235971295824
# DummyRegressor 의 정답률 :  -0.0011144203392519092
# ElasticNet 의 정답률 :  0.1390512965342866
# ElasticNetCV 의 정답률 :  0.6049034437352672
# ExtraTreeRegressor 의 정답률 :  -0.15933833912379902
# ExtraTreesRegressor 의 정답률 :  0.5779155539777245
# GammaRegressor 의 정답률 :  0.08370943030902034
# GaussianProcessRegressor 의 정답률 :  -7.000298067921017
# GradientBoostingRegressor 의 정답률 :  0.5642612076445169
# HistGradientBoostingRegressor 의 정답률 :  0.5452340114658938
# HuberRegressor 의 정답률 :  0.6299216509192822
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 :  0.5422056859784175
# KernelRidge 의 정답률 :  0.6205043348780911
# Lars 의 정답률 :  0.6229230186189662
# LarsCV 의 정답률 :  0.6268464473041173
# Lasso 의 정답률 :  0.5911649816810618
# LassoCV 의 정답률 :  0.6311367325065855
# LassoLars 의 정답률 :  0.43602959555310583
# LassoLarsCV 의 정답률 :  0.6310634453636431
# LassoLarsIC 의 정답률 :  0.6310113874862858
# LinearRegression 의 정답률 :  0.6307176518001982
# LinearSVR 의 정답률 :  0.15917291180251503
# MLPRegressor 의 정답률 :  -0.504164348803382
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 :  0.13087812015152034
# OrthogonalMatchingPursuit 의 정답률 :  0.4323511446277035
# OrthogonalMatchingPursuitCV 의 정답률 :  0.6292810126044563
# PLSCanonical 의 정답률 :  -1.16879940597451
# PLSRegression 의 정답률 :  0.6189994279765481
# PassiveAggressiveRegressor 의 정답률 :  0.6414304236922138
# PoissonRegressor 의 정답률 :  0.605791924480017
# RANSACRegressor 의 정답률 :  -1.5028769402158875
# RadiusNeighborsRegressor 의 정답률 :  0.1583182865249685
# RandomForestRegressor 의 정답률 :  0.5510558943891413
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 :  0.620021551185785
# RidgeCV 의 정답률 :  0.6200215511857821
# SGDRegressor 의 정답률 :  0.6125243474505717
# SVR 의 정답률 :  0.11857410392787959
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 :  0.6141878078202594
# TransformedTargetRegressor 의 정답률 :  0.6307176518001982
# TweedieRegressor 의 정답률 :  0.08533994441547976
# VotingRegressor 은 안나온 놈!!!
#=======================================================================================#


