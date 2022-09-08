import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 - 4*x + 6

x = np.linspace(-1, 6, 100)
print(x, len(x))

y = f(x)


############ 그려!!!
plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')    # 수치상 2, 2는 꼭지점이므로 출력해봄
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()






