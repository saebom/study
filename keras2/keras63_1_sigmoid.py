# 난 정말 시그모이드~~~ !!!

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/ (1 + np.exp(-x))  # 

sigmoid2 = lambda x : 1/ (1 + np.exp(-x))       # np.exp() 함수는 밑이 자연상수 e인 지수함수(e^x)로 변환해줌

x = np.arange(-5, 5, 0.1)   # -5에서 5까지 0.1씩 증가
print(x)
print(len(x))   # 100

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()