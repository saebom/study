import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=66
)


#2. 모델 구성, 훈련, 평가, 예측
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
#ARDRegression 의 정답률 :  0.7834386985676883
# AdaBoostRegressor 의 정답률 :  0.8728171758168148
# BaggingRegressor 의 정답률 :  0.8787324951979545
# BayesianRidge 의 정답률 :  0.7912228365229351
# CCA 의 정답률 :  0.7757272685646834
# DecisionTreeRegressor 의 정답률 :  0.7296787526430877
# DummyRegressor 의 정답률 :  -0.005227869326375867
# ElasticNet 의 정답률 :  0.7364371198416895
# ElasticNetCV 의 정답률 :  0.7213117448627095
# ExtraTreeRegressor 의 정답률 :  0.7913798496930123
# ExtraTreesRegressor 의 정답률 :  0.9071964167330672
# GammaRegressor 의 정답률 :  -0.005227869326375867
# GaussianProcessRegressor 의 정답률 :  -5.81311213590078
# GradientBoostingRegressor 의 정답률 :  0.9138262053434121
# HistGradientBoostingRegressor 의 정답률 :  0.8962100389153277
# HuberRegressor 의 정답률 :  0.6917114545439031
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 :  0.6338244105803171
# KernelRidge 의 정답률 :  0.7871652485288323
# Lars 의 정답률 :  0.8044888426543628
# LarsCV 의 정답률 :  0.8032830033921295
# Lasso 의 정답률 :  0.7234368838497398
# LassoCV 의 정답률 :  0.7579580480062951
# LassoLars 의 정답률 :  -0.005227869326375867
# LassoLarsCV 의 정답률 :  0.8044516427844497
# LassoLarsIC 의 정답률 :  0.7983441148086399
# LinearRegression 의 정답률 :  0.8044888426543619
# LinearSVR 의 정답률 :  0.7298661664984213
# MLPRegressor 의 정답률 :  0.7153578171918873
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 :  0.2886896021336892
# OrthogonalMatchingPursuit 의 정답률 :  0.5651272222459414
# OrthogonalMatchingPursuitCV 의 정답률 :  0.7415292549226284
# PLSCanonical 의 정답률 :  -2.2717245026237816
# PLSRegression 의 정답률 :  0.773871709594815
# PassiveAggressiveRegressor 의 정답률 :  0.22482457867380323
# PoissonRegressor 의 정답률 :  0.8525476757721433
# RANSACRegressor 의 정답률 :  0.5003731681870762
# RadiusNeighborsRegressor 은 안나온 놈!!!
# RandomForestRegressor 의 정답률 :  0.8906441093384034
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 :  0.8031981173042202
# RidgeCV 의 정답률 :  0.8046761789805683
# SGDRegressor 의 정답률 :  -6.440949863823806e+25
# SVR 의 정답률 :  0.26986966510824173
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 :  0.7605913968187908
# TransformedTargetRegressor 의 정답률 :  0.8044888426543619
# TweedieRegressor 의 정답률 :  0.7329785161609925
# VotingRegressor 은 안나온 놈!!!
#=======================================================================================#

