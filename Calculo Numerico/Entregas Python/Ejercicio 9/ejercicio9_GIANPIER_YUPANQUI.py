## EJERCICIO 9
## NOMBRE APELLIDO: Gianpier Yupanqui
## LU 819/18

import numpy.linalg as npl
import numpy as np
import matplotlib.pyplot as plt

# A)
X = np.linspace(0,np.pi,10)
Y = np.sin(X)

# B)
def matriz_A(x,n,tipo):
    nx = len(x)
    A = np.zeros((nx,n)) # nx filas, n columnas
    if tipo=='polinomial':
        for i in range(nx):
            for j in range(n):
                A[i][j] = x[i]**j
    elif tipo=='senoidal':
        for i in range(nx):
            for j in range(n):
                A[i][j] = np.sin((j + 1)*X[i])
    return(A)
    
n = 3
x = np.array([0,2])    
p = matriz_A(x,n,'polinomial')
# array([[1., 0., 0.],
#        [1., 2., 4.]])

s = matriz_A(x,n,'senoidal')

# c)
def cuadrados(x,y,n,tipo):
    A = matriz_A(x,n,tipo)
    B = np.dot(A.T,A)
    c = np.dot(npl.inv(B) , np.dot(np.transpose(A),y))
    return(c)

x= np.array([0,1,2])
y= np.array([0,1,4])
n= 3
tipo='polinomial'
coef = cuadrados(x,y,n,tipo)
print(coef)
"""
B = 3	3	5
    3	5	9
    5	9	17

B*-1 = [ 1. , -1.5,  0.5],
       [-1.5,  6.5, -3. ],
       [ 0.5, -3. ,  1.5]

A.T * y = [5., 9., 17.]    

Numericamente deberia ser igual a (0,0,1) pero hay errores decimales en python
"""

# D)
coef_p = cuadrados(X,Y,n,'polinomial')
coef_s = cuadrados(X,Y,n,'senoidal')

# E)
# -------------------------------------------------
def genera_ajustador(c,tipo):
    if tipo=='polinomial':
        def function(z):
            w = 0
            for j in range(len(c)):
                w += c[j]*(z**j)
            return(w)
    elif tipo=='senoidal':
        def function(z):
            w = 0   
            for j in range(len(c)):
                w += c[j]*np.sin(z*(j+1))
            return(w)
    return(function)

ga = genera_ajustador(coef_p,'polinomial')
gs = genera_ajustador(coef_s,'senoidal')

X_full = np.linspace(0,np.pi,50)
Y_full = np.sin(X)

plt.figure()
plt.title("ajuste_seno")
plt.scatter(X,Y,label="Puntos")
plt.plot(X_full,ga(X_full),label="Polinomio")
plt.plot(X_full,gs(X_full),label="Senos")
plt.grid()
plt.legend()


# F)
# --------------------------------------
import pandas as pd
datos = pd.read_csv('data_prob7.csv') # Por defecto toma 'separador'= (,)

x = np.log2(np.array(datos['area_acres']))
y = np.array(datos['pop_log2'])
tipo='polinomial'

x_full = np.arange(np.min(x),np.max(x),.01)
plt.figure()
plt.title('ajustes_datos')
plt.scatter(x,y,label="puntos")
for n in [2,4,6,8]:
    coef = cuadrados(x,y,n,tipo)
    ga = genera_ajustador(coef,'polinomial')
    plt.plot(x_full,ga(x_full),label="n="+str(n))
plt.legend()

""" El de mayor grado,representa mejor los datos"""

# G Extra 1:
def calcula_error(X,Y,ajustador):
    Y_ajustador = ajustador(X)
    error = npl.norm(Y_ajustador - Y)  # Retorna la norma 2, por defecto
    return(error) # Retorna un numero [UNO]
errores = []

plt.figure()
plt.title('ajustes_datos')
for n in [2,4,6,8]:
    coef = cuadrados(x,y,n,tipo)
    ajustador = genera_ajustador(coef,'polinomial')
    error = calcula_error(x,y,ajustador)
    errores.append(error)
plt.scatter([2,4,6,8],errores)
plt.xlabel('n')
plt.ylabel('error')
plt.legend()

""" > Cual ajuste tiene menor error?
    El de mayor grado
    
    > Se condice con lo que visualmente observo en el punto anterior
    Si porque se va acercando al polinomio interpolador, el cual pasa exactamente por los puntos, teniendo un error de 0
"""

# H Extra 2:
def error_calidad_modelo(X,Y,n,tipo):
    error_valores = []
    for i in range(len(X)):
        X_noi = [xj for j,xj in enumerate(X) if j!= i ] #Se puede mejorar??
        Y_noi = [yj for j,yj in enumerate(Y) if j!= i ]
        c = cuadrados(X_noi,Y_noi,n-1,tipo)
        ajustador = genera_ajustador(c,tipo)
        error = calcula_error(X[i],Y[i],ajustador)
        error_valores.append(error)
    error_medio = np.mean(error_valores)
    return(error_medio)
    
""" Veremos que tambien predice quitando un punto y creando el ajuste con los demas
    Para luego comprobar que tambien predice el punto quitado
"""    
errores = [error_calidad_modelo(x,y,n,'polinomial') for n in [2,4,6,8]]    

plt.figure()
plt.scatter([2,4,6,8],errores,label="puntos")
plt.xlabel('n')
plt.ylabel('error_calidad_modelo')
plt.grid()
plt.legend()    

""" > Que modelo tiene menor error bajo este calculo?
    El n = 4
    
    > El error en el punto G representa, cuan bien nos permite predecir nuevos valores el modelo que consideramos, a partir de
      los datos a los que tenemos acceso.
"""