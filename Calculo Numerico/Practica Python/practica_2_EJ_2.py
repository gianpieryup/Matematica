import numpy as np
import matplotlib.pyplot as plt
from euler import euler

def y_prima(t,y):
    res = (y-5)*(np.cos(t)**2 - 0.5)
    return res

h = 0.01
t0= 0
T = 10
for y0 in range(11):
    vec = euler(y_prima,t0,T,y0,h)
    plt.plot(vec[0],vec[1],label = y0)
plt.xlabel('t')
plt.ylabel('y')
plt.grid()
plt.legend()

