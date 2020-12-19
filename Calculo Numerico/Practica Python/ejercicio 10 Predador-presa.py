"""  Calculo Numerico - UBA-FCEN
Ejercicio 10) Practica 2)          """
import matplotlib.pyplot as plt
import numpy as np
def y_prima(coefs,V):
    """ 
        coefs=[alpha,beta,gamma,delta]
        V = np.array([x,y])
    """
    alpha = coefs[0]
    beta  = coefs[1]
    gamma = coefs[2]
    delta = coefs[3]
    x = V[0]
    y = V[1]

    x_prima = -alpha*x + gamma*x*y
    y_prima = beta*y - delta*x*y
    
    return np.array([x_prima,y_prima])

def euler(coefs,V0,t0,T,h):
    """
        V0 = [x0,y0]
    """
    t = [t0]
    V = [V0]
    while t[-1]<T:
        V_new = V[-1] + h*y_prima(coefs,V[-1])
        t_new = t[-1] + h
        t.append(t_new)
        V.append(V_new)
    return([t,V])

def despliega_integracion(integracion):
    """
       integracion = [lista_t , lista_V]
       lista_V = [[x0,y0],[x1,y1],[x2,y2],...]
    """
    t = integracion[0]
    V = integracion[1]
    X = []
    Y = []
    for v in V:
        X.append(v[0])
        Y.append(v[1])
    return([np.array(t),np.array(X),np.array(Y)])
    
# b)
alpha = 3 
beta  = 5
gamma = 2
delta = 3
coefs = [alpha,beta,gamma,delta]
x0 = beta / delta
y0 = alpha / gamma
V0 = np.array([x0,y0])
t0 = 0
T = 24
h = 0.1

integracion = euler(coefs,V0,t0,T,h)
variables = despliega_integracion(integracion)
t = variables[0]
x = variables[1]
y = variables[2]

plt.plot(t,x,label='Predador')
plt.plot(t,y,label='Presa')
plt.xlabel('Tiempo')
plt.ylabel('Cant animales')
plt.legend()

# c)
alpha = 0.25 
beta  = 1
gamma = 0.01
delta = gamma
coefs = [alpha,beta,gamma,delta]
x0 = 80
y0 = 30
V0 = np.array([x0,y0])
t0 = 0
T = 24
h = 0.1

integracion = euler(coefs,V0,t0,T,h)
variables = despliega_integracion(integracion)
t = variables[0]
x = variables[1]
y = variables[2]
plt.figure()
plt.plot(t,x,label='Predador')
plt.plot(t,y,label='Presa')
plt.xlabel('Tiempo')
plt.ylabel('Cant animales')
plt.legend()


# Implementacion echa en clase)
""" La profe uso esta forma de E.D. un poco distinta (Con los signos cambiados)
def y_prima(coefs,V):
    alpha = coefs[0]
    beta  = coefs[1]
    gamma = coefs[2]
    delta = coefs[3]
    x = V[0]
    y = V[1]

    x_prima = alpha*x - gamma*x*y
    y_prima = -beta*y + delta*x*y
    
    return np.array([x_prima,y_prima])
"""
alpha = 0.4 
beta  = 0.8
gamma = 0.018
delta = 0.023
coefs = [alpha,beta,gamma,delta]
x0 = 30
y0 = 4
V0 = np.array([x0,y0])
t0 = 0
T = 24
n = 400
h = (T-t0)/400

integracion = euler(coefs,V0,t0,T,h)
variables = despliega_integracion(integracion)
t = variables[0]
x = variables[1]
y = variables[2]
plt.figure()
plt.plot(t,x,label='Conejos')
plt.plot(t,y,label='Zorros')
plt.xlabel('Tiempo')
plt.ylabel('Cant animales')
plt.legend()