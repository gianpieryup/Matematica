## EJERCICIO 4
## NOMBRE APELLIDO: GIANPIER YUPANQUI
## LU 819/18

# Ecuacion Diferencial
# S' = -beta.S.I/N
# I' = beta.S.I/N  - gamma.I
# R' = gamma.I

import numpy as np
# N = tamaño de la poblacion
def V_prima(coefs,V):
    """ 
        coefs=[beta,gamma,N]
        V = np.array([S,I,R])
    """
    beta = coefs[0]
    gamma = coefs[1]
    N = coefs[2]
    S = V[0]
    I = V[1]
    R = V[2]
    """
        Para la parte donde agregamos un 'alpha' lo editamos directamente
    """
    alfa = 0  #Editar a mano
    S_prima = -beta*S*I/N+alfa*R
    I_prima = beta*I*S/N-gamma*I
    R_prima = gamma*I-alfa*R
    
    return np.array([S_prima,I_prima,R_prima])

def paso_runge_kutta(coefs,V,h,t):#Modificacion para usar otras V'
    k1 = V_prima(coefs,V)# Es un array
    k2 = V_prima(coefs,V+k1*h/2)#es un array
    k3 = V_prima(coefs,V+k2*h/2)
    k4 = V_prima(coefs,V+k3*h/2)
    V = V + (k1+2*(k2+k3)+k4)/6*h
    t = t+h
    return([t,V])
    
def integra_runge_kutta(coefs,V0,t0,T,h):
    """
        V0 = [S0,I0,R0]
    """
    t = [t0]
    V = [V0]
    while t[-1]<T:
        paso = paso_runge_kutta(coefs,V[-1],h,t[-1])
        t_new = paso[0]
        V_new = paso[1]
        t.append(t_new)
        V.append(V_new)
    return([t,V])
    
def despliega_integracion(integracion):
    """
       integracion = [lista_t , lista_V]
       lista_V = [[S0,I0,R0],[S1,I1,R1],[S2,I2,R2],...]
    """
    t = integracion[0]
    V = integracion[1]
    S = []
    I = []
    R = []
    for v in V:
        S.append(v[0])
        I.append(v[1])
        R.append(v[2])
    return([np.array(t),np.array(S),np.array(I),np.array(R)])
    
    
beta = 1.3  #Jugar con esto o hacer un for mas abajo con distintos beta
gamma = 1
N = 500
coefs=[beta,gamma,N]

S0=N*0.99
I0=N*0.01
R0=0.
V0 = np.array([S0,I0,R0])
    
t0 = 0
T = 50
h = 0.15

integracion = integra_runge_kutta(coefs,V0,t0,T,h)
variables = despliega_integracion(integracion)
t = variables[0]
s = variables[1]
i = variables[2]
r = variables[3]

import matplotlib.pyplot as plt
plt.plot(t,s,label='SUCEPTIBLES', linestyle='--')
plt.plot(t,i,label='INFECTADOS')
plt.plot(t,r,label='RECUPERADOS',linestyle='--')
plt.plot(t,s+r+i,label='S + I + R')
plt.xlabel('Tiempo')
plt.ylabel('Poblacion Total')
plt.legend()

""" > Se mantiene constante en el tiempo? 
    Si vemos que las suma (S +I +R)=10000=N permanece constante atraves del tiempo
"""
L_betas = [0.8,1.5,1.2]
plt.figure()
for b in L_betas:
   coefs=[b,gamma,N] 
   integracion = integra_runge_kutta(coefs,V0,t0,T,h)
   variables = despliega_integracion(integracion)
   t = variables[0]
   i = variables[2] 
   plt.plot(t,i,label='beta: '+str(b))
plt.xlabel('Tiempo')
plt.ylabel('Infectados')
plt.grid()
plt.legend()

""" Mirar todas las graficas
    > Como cambia el valor numero maximo de infectados, asi como el momento en el que se alcanza, para distintos valores de beta? 
    Entre mas grande el 'BETA' se alcanzara mas tempranamente el Momento del PICO
    Entre mas chico 'BETA' < 1.3 aprox
    BETA > 1 el momento del PICO se alcanza mas Tarde 
    
    > Que pasa si 'beta' < 'gamma'
    EL PICO se alcanza en el momento 0, osea solo diminuye (desde la cant inicial)
    
    > Siempre se llega a infectar a toda la poblacion?
    Vemos que el Pico Maximo crece - a medida que Beta crece
    
    Se esta considerando (gamma = 1)
"""
t_max = []
I_max = []
betas = np.arange(.9,5,.05)
for beta in betas:
    coefs = [beta,gamma,500]
    integracion = integra_runge_kutta(coefs,V0,t0,T,h)
    variables = despliega_integracion(integracion)
    I = variables[2] # Pos donde estan solo los valores de los INFECTADOS
    t_max.append(np.argmax(I)*h) # La posicion del maximo valor de la lista de INFECTADOS
    """ Si V=[10,20,30,20,10,10] => np.argmax(V) := 2
        Pero como t0=0, t1 = h+t0 ,... ,tn =h*n        """
    I_max.append(np.max(I)) # El maximo valor de la lista de INFECTADOS


# MOMENTO del Pico Maximo
plt.figure()
plt.plot(betas,t_max)
plt.xlabel('Beta')
plt.ylabel('Momento del pico')

# VALOR del Pico Maximo
plt.figure()
plt.plot(betas,I_max)
plt.xlabel('Beta')
plt.ylabel('Pico máximo')


"""
    Para probar con 'alpha' distintos cambiar el valor a mano, en la funcion "y_prima"
    alpha = 0.3

    > Que efecto tiene en la evolucion?
    Llegara un punto en que Los infectados y los recuperados, no varian en el tiempo
"""