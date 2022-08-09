import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
# print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

df = pd.DataFrame(x, columns=[['sepal length', 'sepal width', 'petal length', 'petal width']])

df['target(y)'] = y

print(df) # [150 rows x 5 columns]  sklearn의 데이터를 DataFarame으로 바꿔 줌

print('========================= 상관계수 히트 맵 =============================')
print(df.corr())    # 상관관계 확인

# ========================= 상관계수 히트 맵 =============================
#              sepal length sepal width petal length petal width  target(y
# sepal length     1.000000   -0.117570     0.871754    0.817941  0.782561
# sepal width     -0.117570    1.000000    -0.428440   -0.366126 -0.426658
# petal length     0.871754   -0.428440     1.000000    0.962865  0.949035
# petal width      0.817941   -0.366126     0.962865    1.000000  0.956547
# target(y         0.782561   -0.426658     0.949035    0.956547  1.000000

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 1.2)
sns.set(rc={'figure.figsize':(9, 6)}) # 가로세로 사이즈 세팅
sns.heatmap(data=df.corr(), # 상관관계
            square=True,    # 정사각형으로 view
            annot=True,     # 각 cell의 값 표기 유무
            cbar=True       # colorbar의 유무
            )

plt.show()

