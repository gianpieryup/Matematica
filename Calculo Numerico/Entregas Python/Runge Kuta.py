# Ejemplo de resolución, pendulo con disipación

# En este caso, tomo V = [o,w]
# o' = w
# w' = -Asin(o) -d*w
import numpy as np
def V_prima(coefs,V):
    A = coefs[0]
    d = coefs[1]
    o = V[0]
    w = V[1]
    resultado = np.array([w,-A*np.sin(o)-d*w])
    return(resultado)

def paso_runge_kutta(coefs,V,h,t):
    k1 = V_prima(coefs,V)
    k2 = V_prima(coefs,V+k1*dt/2)
    k3 = V_prima(coefs,V+k2*dt/2)
    k4 = V_prima(coefs,V+k3*dt)
    V = V + (k1+2*(k2+k3)+k4)/6*h
    t = t + h
    return([t,V])

def integra_runge_kutta(coefs,V0,t0,T,dt):
    t = [t0]
    V = [V0]
    while t[-1]<T:
        V_old = V[-1]
        t_old = t[-1]
        paso = paso_runge_kutta(coefs,V_old,dt,t_old)
        t_new = paso[0]
        V_new = paso[1]
        t.append(t_new)
        V.append(V_new)
    return([t,V])

def despliega_integracion(integracion):
    t = integracion[0]
    V = integracion[1]
    o = []
    w = []
    for i in range(len(V)):
        o.append(V[i][0])
        w.append(V[i][1])
    return([np.array(t),np.array(o),np.array(w)])

A = 10
d = 5*10**-1
coefs = [A,d]
    
t0 = 0
T = 10
dt = 0.01
o0 = np.pi/4
w0 = 0
V0 = np.array([o0,w0])

integracion = integra_runge_kutta(coefs,V0,t0,T,dt)
variables = despliega_integracion(integracion)
t = variables[0]
o = variables[1]
w = variables[2]

import matplotlib.pyplot as plt
plt.plot(t,o)
plt.xlabel('Tiempo')
plt.ylabel('Theta')
