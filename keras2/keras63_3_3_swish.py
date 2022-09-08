import numpy as np
import matplotlib.pyplot as plt

def swish(x, a):
    1/(1+np.exp(-x))*x
    
swish2 = lambda x : 1/(1+np.exp(-x))*x  
   
x = np.arange(-5, 5, 0.1)   # -5에서 5까지 0.1씩 증가
y = swish2(x)

plt.figure(figsize=(8,5))
plt.plot(x,y,label='swish')
plt.hlines(0,-5,5)
plt.vlines(0,-1,5)
plt.title("Swish",fontsize=24)
plt.xlabel('x',fontsize=24)
plt.ylabel('y',fontsize=24)
plt.grid(alpha=0.3)
plt.legend(loc='upper left',fontsize=15)
plt.show()
