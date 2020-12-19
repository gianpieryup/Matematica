import numpy as np
import matplotlib.pyplot as plt
from euler import euler

def tao(t,y): return y
def tao1(t,y): return 10*y
def tao2(t,y): return 50*y
def tao3(t,y): return 100*y
    
# considero
t0 = 0
T = 1
y0 = 1
h = 0.5
vec = euler(tao,t0,T,y0,h)
vec1 = euler(tao1,t0,T,y0,h)
vec2 = euler(tao2,t0,T,y0,h)
vec3 = euler(tao3,t0,T,y0,h)

    
plt.plot(vec[0],vec[1],label = 'lam = 1')
plt.plot(vec1[0],vec1[1],label = 'lam = 10')
plt.plot(vec2[0],vec2[1],label = 'lam = 50')
plt.plot(vec3[0],vec3[1],label = 'lam = 100')
plt.grid()
plt.legend()

#Con euler implicito
t0 = 0
T = 1
y0 = 1
h = 0.25
def euler_implicito(t0,T,y0,h,l):
    t = [t0]
    y = [y0]
    N = int((T-t0)/h)
    for c in range(N):#son N operaciones
        yN = y[-1]/(1-h*l)
        tN = t[-1] + h
        t.append(tN)
        y.append(yN)
    return [t,y]

for l in [1,10,50,100]:
    vec = euler_implicito(t0,T,y0,h,l)
    plt.plot(vec[0],vec[1],label = l)

plt.grid()
plt.legend()
