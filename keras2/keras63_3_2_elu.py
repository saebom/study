import numpy as np
import matplotlib.pyplot as plt

def elu(x, alp):
    (x>0)*x+ (x<=0)*(alp*(np.exp(x)-1))
    
elu2 = lambda x, alp=1 :(x>0)*x+ (x<=0)*(alp*(np.exp(x)-1))
   
x = np.arange(-5, 5, 0.1)   # -5에서 5까지 0.1씩 증가
y = elu2(x)


plt.figure(figsize=(8,5))
plt.plot(x,y)
plt.hlines(0,-5,5)
plt.vlines(0,-1,5)
plt.title("ELU",fontsize=24)
plt.xlabel('x',fontsize=24)
plt.ylabel('y',fontsize=24)
plt.grid(alpha=0.3)
plt.legend(loc='upper left',fontsize=15)
plt.show()