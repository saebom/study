import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)   # -5에서 5까지 0.1씩 증가
y = np.tanh(x)

plt.plot(x, y)
plt.grid()
plt.show()