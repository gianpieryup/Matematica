## EJERCICIO 6
## NOMBRE APELLIDO: Gianpier Yupanqui
## LU 819/18
import numpy as np
import numpy.linalg as npl

A1 = np.array([[1, 0],[0, 2]])
A2 = np.array([[1, 1],[0, 2]])
b = [1, 1]

#A)
x1 = np.dot(npl.inv(A1),b)
x2 = np.dot(npl.inv(A2),b)

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
    B = -1*np.dot(npl.inv(M),N)
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

#C)
x0 = [3,4]
max_iter = 20
tol = 1
jacobi_A1 = jacobi(A1,b,x0,max_iter,tol) 
print("x1 =",x1,"| jacobi =",jacobi_A1[0],"cant_rep :",jacobi_A1[1])
 
jacobi_A2 = jacobi(A2,b,x0,max_iter,tol)
print("x2 =",x2,"| jacobi =",jacobi_A2[0],"cant_rep :",jacobi_A2[1])


#D)
def gauss_siegel(A,b,x0,max_iter,tol):
    """ A.x= b | una semilla x0 
        max_iter : cant maximo de iteraciones
        tol : error minimo aceptable
    """
    D = np.diag(np.diag(A))
    L = np.tril(A,k=-1)
    U = np.triu(A,k=1)
    M = D + L
    N = U
    B = -1*np.dot(npl.inv(M),N)
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

#F)    
gauss_siegel_A1 = jacobi(A1,b,x0,max_iter,tol) 
print("x1 =",x1,"| GS =",gauss_siegel_A1[0],"cant_rep :",gauss_siegel_A1[1])
 
gauss_siegel_A2 = jacobi(A2,b,x0,max_iter,tol)
print("x2 =",x2,"| GS =",gauss_siegel_A2[0],"cant_rep :",gauss_siegel_A2[1])    

#G)
def matriz_y_vector(n):
    A = np.random.rand(n, n)
    while npl.det(A)==0:
        A = np.random.rand(n, n)
    b = np.random.rand(n,1) # Yo me etivoque aca le puse (n) sin el 1 . Esto gerera un vector 1x3 y no 3x1
    return([A,b])
""" El error anterior, hacia muy lento las iteraciones, tipo en mas de 1 min no llegaba a la iteracion 100 """


#H)
trials_J = []
trials_GS = []
Ntrials = 10**4
trials = 0
x0 = np.transpose(np.array([[0,0,0]]))
max_iter = 10**6  #Para que se demore menos cuando no alcanza la cota
tol = 10**-5
while trials<Ntrials:
    print("iteracion :",trials)
    Ab = matriz_y_vector(3)
    A = Ab[0]
    b = Ab[1]
    rGS= gauss_siegel(A,b,x0,max_iter,tol)
    rJ = jacobi(A,b,x0,max_iter,tol)
    if rJ[1]!=max_iter and rGS[1]!=max_iter:  # Para no tener los casos donde no se llega al error deseado
        trials_J.append(rJ[1])
        trials_GS.append(rGS[1])
        trials = trials + 1
        print(trials)

import matplotlib.pyplot as plt
plt.scatter(trials_J,trials_GS)
plt.ylabel('Gauss-Siegel')
plt.xlabel('Jacobi')
plt.yscale('log')#OSEA ESCALA 10*n
plt.xscale('log')#OSEA ESCALA 10*n 
plt.plot([1,10**6],[1,10**6],color='red') # Hacer una recta
plt.grid()
plt.savefig('grafica_ej6.png')
""" ¿que puede decir sobre el desempeño de ambos metodos?
    para los datos vemos una tendencia
    En general el numero de iteraciones de Gauss-Seidel es menor que el de Jacobi
    Pero hay situaciones donde esto no cumple. muchas    
"""