import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x):
    np.maximum(0.1*x, x)  
    
leaky_relu2 = lambda x : np.maximum(0.1*x, x)   
    
x = np.arange(-5, 5, 0.1)   # -5에서 5까지 0.1씩 증가
y = leaky_relu2(x)

plt.figure(figsize=(8,5))
plt.plot(x,y,label='Leaky ReLU')
plt.hlines(0,-5,5)
plt.vlines(0,-1,5)
plt.title("Leaky ReLU",fontsize=24)
plt.xlabel('x',fontsize=24)
plt.ylabel('y',fontsize=24)
plt.grid(alpha=0.3)
plt.legend(loc='upper left',fontsize=15)
plt.show()