## EJERCICIO 10
## NOMBRE APELLIDO: Gianpier Yupanqui
## LU 819/18

import numpy.linalg as npl
import numpy as np
import matplotlib.pyplot as plt

# A
# ----------------------------
# Los ceros de "f" son [-2, -1]
def funcion_f(x): return (x + 2)*(x + 1)
def funcion_fprima(x): return 2*x + 3

# B
# --------------------------------
def iterador_newton_rhapson(funcion_g,funcion_gprima,x0,max_iter,tol):
    """ funcion_g(xn) -> 0
        Usamos asi el error, porque no sabemos la raiz "La idea es calcularla"
    """
    error = abs(funcion_g(x0))
    iter = 0
    while max_iter > iter and error>tol:
        x1 = x0 - funcion_g(x0)/funcion_gprima(x0)
        error = abs(funcion_g(x0))
        iter = iter+1
        x0 = x1
    return([x0,iter,error])    
    
x0=1  # Se va a la raiz de -1
res = iterador_newton_rhapson(funcion_f,funcion_fprima,x0,15,10**-5)
print("xn=",res[0],"| iteraciones",res[1],"| error",res[2])

x0=-10  # Se va a la raiz de -2
res = iterador_newton_rhapson(funcion_f,funcion_fprima,x0,15,10**-5)
print("xn=",res[0],"| iteraciones",res[1],"| error",res[2],"\n")  
    

# C
# -----------------
def iterador_newton_rhapson(fun_f,fun_fprima,x0,max_iter,tol):
    if fun_fprima=='NO':
        def funcion(x):
            x1 = x*1.01 # Por ejemplo, aproximo usando el 100.1%
            x0 = x*0.99 # y el 99.9% de x
            f1 = fun_f(x1)
            f0 = fun_f(x0)
            f_prima = (f1 - f0) / (x1 - x0)
            return(f_prima)
        
        fun_fprima= funcion
    error = abs(fun_f(x0))
    iter = 0
    while max_iter > iter and error>tol:
        x1 = x0 - fun_f(x0)/fun_fprima(x0)
        error = abs(fun_f(x0))
        iter = iter+1
        x0 = x1
    return([x0,iter,error])

x0=1  # Se va a la raiz de -1
res = iterador_newton_rhapson(funcion_f,"NO",x0,15,10**-3)
print("xn=",res[0],"| iteraciones",res[1],"| error",res[2])

x0=-10  # Se va a la raiz de -2
res = iterador_newton_rhapson(funcion_f,"NO",x0,15,10**-3)
print("xn=",res[0],"| iteraciones",res[1],"| error",res[2],"\n")  

# Para no tener  "fun_fprima" y obtenerla aproximada numericamente, funciona muy bien


# D
# -------------------
def iterador_punto_fijo(fun_g,x0,max_iter,tol):
    """ g(r) = r
    """
    error = abs(fun_g(x0)-x0)
    iter = 0
    while max_iter >= iter and error>tol:
        x1 = fun_g(x0)
        error = abs(x1-x0)
        iter = iter + 1
        x0 = x1
    return([x0,iter,error]) 

def g1(x): return -2/(x + 3)
def g2(x): return -(x**2 + 2)/3
# cumplen que g1,2(x)=x -> f(x) = 0

print("--------  Punto Fijo  -------")
# Se va a la raiz de -1
res = iterador_punto_fijo(g1,1,10**4,10**-5)
print("Con g1")
print("xn=",res[0],"| iter",res[1],"| error",res[2])

# Se va a la raiz de -1
res = iterador_punto_fijo(g2,1,10**4,10**-5)
print("Con g2")
print("xn=",res[0],"| iter",res[1],"| error",res[2],"\n")

""" Newton-R converge mas rapido porque aprovecha la funcion_prima
    Punto Fijo no, es mas lento. Pero no nesecito saber funcion_prima para la iteracion
"""


# E
# --------------------------
x0=1
x = np.arange(0,5.5,.5)
y = np.array([0.756,0.561,0.407,0.372,0.305,0.24,0.219,0.209,0.21,0.194,0.140])


def g(b): 
    sumatoria = 0
    for i in range(1,11):
        sumatoria += (y[i] - 1/(x[i] + b))*((x[0]+b)/(x[i]+b))**2
    return 1/(y[0] + sumatoria) - x[0]


def f(b):
    sumatoria = 0
    for i in range(11):
        sumatoria += (y[i] - 1/(x[i] + b))*(1/(x[i]+b))**2
    return sumatoria

x_full = np.linspace(0,5,100)

res = iterador_newton_rhapson(f,"NO",x0,10**5,.001)
print("Newton-Rap b =",res[0])
def F_new(x): return 1/(x + res[0])
y_new = F_new(x_full)

res = iterador_punto_fijo(g,1,10**5,.001)
print("Punto fijo b =",res[0])
def F_punto_Fijo(x): return 1/(x + res[0])
y_puntoF = F_punto_Fijo(x_full)

plt.scatter(x,y,label="puntos")
plt.plot(x_full,y_new,label="NR")
plt.plot(x_full,y_puntoF,label="PuntoF")
plt.legend()

# F-EXTRA
# ------------
""" Lo comparo con el mejor polinomial (el de grado 4) que es el que tubo menor error_calidad_modelo
"""
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

def cuadrados(x,y,n,tipo):
    A = matriz_A(x,n,tipo)
    B = np.dot(A.T,A)
    c = np.dot(npl.inv(B) , np.dot(np.transpose(A),y))
    return(c)

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

# -----------
import pandas as pd
datos = pd.read_csv('data_prob7.csv')    
X = np.log2(np.array(datos['area_acres']))
Y = np.array(datos['pop_log2'])

X_norm = (X-np.min(X))/(np.max(X)-np.min(X))
Y_norm = (Y-np.min(Y))/(np.max(Y)-np.min(Y))

# Primero hallar el polinomio interpolador
# Usare Punto Fijo
"""
def F_prima(a): 
    sumatoria = 0
    for i in range(1,11):# del 1 al 10 || el 0 y el 11 solo suman 0
        sumatoria += (Y_norm[i] - X_norm[i]**a)*(-a*X_norm[i]**(a-1))
    return sumatoria
[MALLLL] LA derivada se hace en [a] no en x? ni siquiera hay x, solo x0's
 La derivada de x**a [en a] ->    
"""
def F_prima(a):
    x = X_norm[1:]
    y = Y_norm[1:]
    result = np.sum((y-x**a)*np.log(x)*x**a)
    return(result)


res = iterador_newton_rhapson(F_prima,"NO",1,10**5,.01)[0] #VALOR DE a

#-------Cuadrado Minimo -----------
coef = cuadrados(X_norm,Y_norm,4,'polinomial')
ga = genera_ajustador(coef,'polinomial')

#----------------------------------

plt.figure()
plt.scatter(X_norm,Y_norm,label="puntos",color="red")
plt.plot(X_norm,X_norm**res,label="NR")  # La funcion es x**a
plt.plot(X_norm,ga(X_norm),label="Pol grado 4",color="green")
plt.legend()