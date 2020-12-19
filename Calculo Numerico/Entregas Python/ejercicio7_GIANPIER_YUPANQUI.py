## EJERCICIO 7
## NOMBRE APELLIDO: Gianpier Yupanqui
## LU 819/18

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
# En la definicion de P le falta el 0

#A)
def pol_i(x,i,m):
    res = 1
    j=0
    while j < i:
        res *= m - x[j]
        j+=1
    return res

def matriz_dif(x):
    A = np.zeros((len(x),len(x)))
    A[:,0] = 1
    for q in range(len(x)):
        for r in range(1,q+1):
            A[q,r] = pol_i(x,r,x[q])  
    return(A)
    
#x = [3,5,8,12]  # productoria(x,q,r)
#matriz_dif(x) 

#B)
def jacobi(A,b,x0,max_iter,tol):
    """ A.x= b | una semilla x0 
        max_iter : cant maximo de iteraciones
        tol : error minimo aceptable
    """
    D = np.diag(np.diag(A))
    L = np.tril(A,k=-1)
    U = np.triu(A,k=1)
    M = D
    N = U + L
    B = -np.dot(npl.inv(M),N)
    C = np.dot(npl.inv(M),b)
    xN = x0
    resto = npl.norm(np.dot(A,xN)- b)
    itera = 0

    while resto > tol and itera < max_iter:        
        xN = np.dot(B,xN) + C
        itera = itera + 1
        resto = npl.norm(np.dot(A,xN)- b)
        
    if itera==max_iter:
        print('Max_iter reached')
    return([xN,itera])
    
def newton_dif(x,y,max_iter,tol):
    A = matriz_dif(x)
    b = y
    x0 = np.zeros(len(y))
    coef = jacobi(A,b,x0,max_iter,tol)
    return(coef)
    
    
# C)
# Devuelve la funcion que cumple f(x_i) = y_i 
def funcion_interpol(x,y,max_iter,tol):
    coef = newton_dif(x,y,max_iter,tol)[0]
    def function(m):
        X = np.ones(len(coef))
        for i in range(len(coef)):
            X[i] = coef[i]*pol_i(x,i,m)
        return sum(X)
    return(function)    
    
#D)
tol = 10**-3
max_iter = 10**6
x = np.linspace(0,np.pi,10) # Puntos a inerpolar
y = np.sin(x) # Puntos a inerpolar

x_full = np.linspace(0,np.pi,32) # Muchos puntos
fun_int = funcion_interpol(x,y,max_iter,tol)
y_int = np.array([fun_int(xi) for xi in x_full])

plt.scatter(x,y,label="funcion")  # 10 Puntitos
plt.plot(x_full,y_int,label="interpolado")
plt.legend()

""" Notamos que la funcion interpolada, se comporta como deberia un 'SEN(x)'
"""

#D)
plt.figure()
import pandas as pd
datos = pd.read_csv('data_prob7.csv') # Por defecto toma 'separador'= (,)
x = np.log2(np.array(datos['area_acres']))
y = np.array(datos['pop_log2'])

x_full = np.arange(np.min(x),np.max(x),.01) #Tomamos unos puntos de prueba, de preferencia muchos
y_inter = []

f = funcion_interpol(x,y,max_iter,tol) #NO CONFUNDIR LOS x y con // los x_full e y_inter
y_inter = np.array([f(xi) for xi in x_full])

plt.scatter(x,y,label="datos") # 12 Puntitos
plt.plot(x_full,y_inter,label="interpolado")
plt.legend()    

"""
    El polinomio interpolado oscila muchas veces, cuando se puede encontrar un polinomio de menor grado (o que no lo haga tanto)
    plt.scatter  / OSEA SOLO PUNTITOS
    plt.plot     / UNA GRAFICA, DE LA UNION DE LOS PUNTOS
"""