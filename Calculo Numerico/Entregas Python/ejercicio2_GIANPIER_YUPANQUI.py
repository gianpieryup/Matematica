## EJERCICIO 2
## NOMBRE APELLIDO: GIANPIER YUPANQUI
## LU 819/18


import numpy as np
import matplotlib.pyplot as plt

def esPrimo(n):
    if n==1 :
        return False
    for i in range(2,n):
        if(n%i==0):
            return False
    return True

def primos_hasta(N):
    res = []
    for i in range(1,N+1):
        if esPrimo(i):
            res.append(i)
    return res

def cuantos_primos(N,L):#Supongamos que L es un vector de numeros primos
    #Devuelve la cant de primos de L, que son menores o iguales que N
    return  np.sum(np.array(L)<=N) 

N = 10000
L = primos_hasta(N)
x = np.arange(1,N+1)
y = []
for xi in x:
    # En este for, anotamos cuantos primos hay para cada valor xi
    y.append(cuantos_primos(xi,L))

plt.plot(x,y,label = 'Cantidad primos')

b = 200
a=.12
plt.plot(x,a*x+b,label='LINEAL')

b = 200
a=20
plt.plot(x,a*np.sqrt(x)+b,label='RAIZ CUADRADA')

b = 200
a=100
plt.plot(x,a*np.log(x)+b,label='LOGARITMO')

plt.xlabel('Hasta')
plt.ylabel('Cantidad de primos')
plt.grid()      # Para que salga con grilla
plt.legend ()   # Para que funcione el label