import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    np.maximum(0, x)    # 최대값을 잡아라, 0이하이면 0으로 나머지 최대값
    
relu2 = lambda x : np.maximum(0, x)    
    
x = np.arange(-5, 5, 0.1)   # -5에서 5까지 0.1씩 증가
y = relu2(x)

plt.plot(x, y)
plt.grid()
plt.show()

# [실습]
# 3_2, 3_3, 3_4 
# elu, swish, reaky relu
