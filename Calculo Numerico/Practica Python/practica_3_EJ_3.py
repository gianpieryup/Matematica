import numpy.linalg as npl
import numpy as np
import matplotlib.pyplot as plt

def matriz_evolucion(N,k,h,m):
    resultado = []
    for q in range(N):
        fila = []
        for r in range(N):
            if q==r: # Si estoy en la diagonal
                fila.append(2 - k*h**2/m)
            elif q==r-1 or q==r+1: # Si estoy en 1 menos que la diagonal o 1 mas
                fila.append(-1)
            else: # Si no es ningun caso
                fila.append(0)
        resultado.append(fila)
    return(np.array(resultado))
    
    
yf = 1
t0 = 0
tf = 10
m = 1/4
k = 1/2
h = 0.5
N = int((tf-t0)/h)

M = matriz_evolucion(N,k,h,m)
b = np.zeros(N)
b[-1] = yf   # yf = 1 dato
t = [n*k for n in range(N)]
res = np.dot(npl.inv(M),b)
plt.plot(t,res,label="m:1/4,k:1/2")    
###
m = 0.025
M = matriz_evolucion(N,k,h,m)
res = np.dot(npl.inv(M),b)
plt.plot(t,res,label="m:0.025,k:1/2")

m = 1/4
k = 0.05
M = matriz_evolucion(N,k,h,m)
res = np.dot(npl.inv(M),b)
plt.plot(t,res,label="m:1/4,k:0.05")

m = 0.025
M = matriz_evolucion(N,k,h,m)
res = np.dot(npl.inv(M),b)
plt.plot(t,res,label="m:0.025,k:0.05")
plt.legend()
"""
Cuando (k/m) es igual, se generan las mismas matrices por ende los mismos resultados
"""