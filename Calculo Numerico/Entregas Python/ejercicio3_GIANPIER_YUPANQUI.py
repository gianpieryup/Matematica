## EJERCICIO 3
## NOMBRE APELLIDO: GIANPIER YUPANQUI
## LU 819/18

import matplotlib.pyplot as plt
import numpy as np
"""-------------------  PARTE A  --------------------------"""
# Partimos de la ecuación
## y' = -2y + t**2
## y0 = 1

def y_exacta(t):
    return 0.75*(np.e**(-2*t)) + 0.5*(t**2 - t + 0.5)

def y_prima(t,y,h):# Construyo y_prima osea (y')
    return -2*y + t**2

def resuelve_y(y0,t0,T,N):
    t = [t0]
    y = [y0]
    h = (T-t0)/N
    for c in range(N):#son N operaciones
        yN = y[-1] + h*(y_prima(t[-1],y[-1],0))
        tN = t[-1] + h
        t.append(tN)
        y.append(yN)
    return [np.array(t),np.array(y)]#OJO no [t,y]

t0 = 0
T = 10
y0 = 1
N = [10,50,100,500,1000]
errores = []
resultados = []
for n in N:
    vec = resuelve_y(y0,t0,T,n)
    err = abs(y_exacta(10) - vec[1][-1])
    resultados.append(vec)
    errores.append(err)
    plt.plot(vec[0],vec[1],label=n)
    

tE = np.arange(0,T+1,1)
yE = y_exacta(tE)
plt.plot(tE,yE,label='EXACTA')
plt.xlabel('Tiempo')
plt.ylabel('y')
plt.grid()
plt.legend()
#plt.savefig('pa_evoluciony.png')

# Error en general (y-yE) en funcion del 'Tiempo'
plt.figure()#Comando para decir que sigue una nueva grafica
for i in range(len(N)):
    resultado = resultados[i] # agarro el resultado i-esimo
    n = N[i]
    t = resultado[0]
    y = resultado[1]
    yE = y_exacta(t)
    plt.plot(t,abs(y-yE),label=n)    
plt.xlabel('Tiempo')
plt.ylabel('y-yE')
plt.legend()
#plt.savefig('pa_error_general.png')

# Error en T=10 en funcion a 1/N (1/cantidad de pasos)
plt.figure()#Comando para decir que sigue una nueva grafica
plt.plot(1/np.array(N),errores,label='ERROR')
plt.xlabel('1/N')
plt.ylabel('y(T)-yE(T)')
plt.legend()



"""---------------------  PARTE B  ------------------------"""

N = 100 #Probar con 100-200-500
t0 = 0
T = 10 #En la solucionn usa 30 para ver como sigue el comportamiento 

def Euler_A(t0,T,N):
    y1 = [0]
    y2 = [1]  #y1 e y2 son las listas con los valores iniciales 
    t = [t0]
    h = (T-t0)/N
    for c in range(N):
        y1_N = y1[-1] + h* y2[-1]
        y2_N = y2[-1] + h*(-2* y1[-1])
        tN = t[-1] + h
        t.append(tN)
        y1.append(y1_N)
        y2.append(y2_N)
    return [t,y1,y2]
EA = Euler_A(t0,T,N)

def Euler_B(t0,T,N):
    y1 = [0]#los pongo aca si los paso como parametro se ,los cambios en la lista persiste
    y2 = [1]
    t = [t0]
    h = (T-t0)/N
    for c in range(N):
        y1_N = y1[-1] + h*y2[-1] 
        y2_N = y2[-1] + h*(-2*y1_N)         
        tN = t[-1] + h
        t.append(tN)
        y1.append(y1_N)
        y2.append(y2_N)
    return [np.array(t),np.array(y1)] #No nos importa guardar y2
EB = Euler_B(t0,T,N)

def Euler_C(t0,T,N):
    y1 = [0]#los pongo aca si los paso como parametro se ,los cambios en la lista persiste
    y2 = [1]
    t = [t0]
    h = (T-t0)/N
    for c in range(N):
        y2_N = y2[-1] + h*(-2* y1[-1]) 
        y1_N = y1[-1] + h* y2_N           
        tN = t[-1] + h
        t.append(tN)
        y1.append(y1_N)
        y2.append(y2_N)
    return [np.array(t),np.array(y1)]

EC = Euler_C(t0,T,N)

plt.figure()#Comando para decir que sigue una nueva grafica
plt.plot(EA[0], EA[1], label = "EULER A")
plt.plot(EB[0], EB[1], label = "EULER B")
plt.plot(EC[0], EC[1], label = "EULER C")

def y_solucion_2(t):
    return (1/np.sqrt(2))*np.sin(np.sqrt(2)*t)

tE = np.arange(0,T+1,1)
yE = y_solucion_2(tE)
plt.plot(tE,yE,label='EXACTA')
plt.grid()
plt.legend()


# Elegimos por ejemplo método B
# Y hacemos un barrido en N
plt.figure()
N_lista = np.array([100,500,1000])
resultados = []
for n in N_lista:
    vec = Euler_B(t0,T,n)
    t = vec[0]
    y = vec[1]
    yE = y_solucion_2(t)
    plt.plot(t,abs(y-yE),label=n)    
plt.xlabel('Tiempo')
plt.ylabel('Error')
plt.legend()
#plt.savefig('pb_errort.png')


""" ¿Como aumenta el error en funcion del numero de ciclos, para cada N?
    Entre mas ciclos (+N), menor es el error . Osea las aproximaciones[EULER] se acercan mas a la solucion original
    Yo probe con N = 100,200 y 500 solo cambiar al inicio y volver a ejecutar
    
    ¿Respeta el periodo la solucion? 
    SI, mira la grafica
"""


"""-------------------  PARTE C  --------------------------"""
def resuelve_y(y0,t0,T,N,y_prima):
    t = [t0]
    y = [y0]
    h = (T-t0)/N
    for c in range(N):#son N operaciones
        yN = y[-1] + h*y_prima(t[-1],y[-1],0)
        tN = t[-1] + h
        t.append(tN)
        y.append(yN)
    return [t,y]
#Basicamente es el ejercicio 4 de la guia 2