import numpy as np
import matplotlib.pyplot as plt
# Integramos entre a y b, usando cuadraturas compuestas (partirmos el intervalo en n pedacitos,
# e integramos con simpson o trapecios en cada intervalo)


# Método de trapecios:

def q_trapecios(f,a,b,n):
    q = 0 # arranco la cuadratura en 0
    paso = (b-a)/n # Este es el paso de mi intervalo
    x0 = a #Voy evaluando f en el principio y final del intervalo
    f0 = f(x0)
    while x0<b: # Hasta llegar a b
        x1  = x0+paso # El segundo punto
        f1 = f(x1)
        q += (f1+f0)/2*paso # Sumo la cuadratura de ese paso
        x0 = x1 # Actualizo valores
        f0 = f1
    return(q)

# Método de simpson
## Igual a trapecios, pero haciendo simpson
def q_simpson(f,a,b,n):
    q = 0 
    paso = (b-a)/n
    h = paso/2 # Acá tenemos 2 pasos
    x0 = a
    f0 = f(x0)
    while x0<b:
        x1  = x0+paso
        xh = x0+h
        f1 = f(x1)
        fh = f(xh)
        q += h/3*(f0+4*fh+f1)
        x0 = x1
        f0 = f1
    return(q)



# Probemoslo!
# Defino el intervalo
a = 0
b = 1


def f(x):
    result = np.exp(x)
    return(result)

ns = np.arange(1,100) # Usamos n entre 1 y 100
# Ya sabemos cuanto tiene que dar: int^1_0 exp(x) dx = exp(1)-exp(0) = e-1
trape = [np.e -1 - q_trapecios(f,a,b,ni) for ni in ns]
simpson = [np.e -1 - q_simpson(f,a,b,ni) for ni in ns]

# Grafico
plt.clf()
plt.plot(ns,trape,label='Trapecios') #  tienen el mismo grafica[error]
plt.plot(ns,simpson,label='Simpson') #  tienen el mismo grafica[error]
# Pareciera caer como 1/n el error
plt.plot(ns,-1/ns,label='Tendencia?')


# Cuadratura gaussiana
# Usamos las funciones de numpy

from scipy import integrate as sint

Qgauss = sint.quadrature(f,0,1) # Retorna dos valores: el resultado y un estimado del error
plt.axhline(y=np.e-1-Qgauss[0],label='Gauss')
plt.legend()
plt.savefig('cuadraturas.png')