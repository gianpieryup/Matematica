import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from interpolacion import newton_dif  # Para usar lasl interpolado

""" Ej 1 - 2 """
def cuadrados_minimos(x,y,n):
    A = []
    for xi in x:
        fila = []
        for j in range(n + 1):
            fila.append(xi**j)
        A.append(fila)
    #print(A)
    A_ = np.dot(np.transpose(A),A)  # A.T si A no es una matriz, no puedo
    b = np.dot(np.transpose(A),y)
    q , r = npl.qr(A_) #descomposicion QR
    coef = np.dot(npl.inv(r) , np.dot(q.T,b))
    print(coef)
    def function(m):
        X = np.ones(len(coef))
        for i in range(len(coef)):
            X[i] = coef[i]*(m**i)
        return sum(X)
    return(function)    

x = [ 60.3, 37.4, 65, 74.8, 39, 21.1, 36.1, 38.6, 53.1, 65.7, 16.8, 32.4, 78.8, 15.3, 84.8, 71.6, 54.7, 67.1, 35, 49.7, 80.5, 66.8, 60.4, 30.7, 50.2]
y = [ 8, 6, 8, 9, 6, 5, 6, 6, 7, 8, 5, 6, 9, 5, 10, 9, 7, 8, 6, 7, 10, 8, 8, 6, 7]

""" Lineal """
f1 = cuadrados_minimos(x,y,1)
x_full = np.arange(0,100,0.5)
f_x_full = [f1(xi) for xi in x_full]

plt.figure()
plt.scatter(x,y,label="puntos")
plt.plot(x_full,f_x_full,label="cuadrado minimos")
plt.plot([0,100],[8,8],label="8 final")
plt.grid()
plt.legend()

""" Cuadratica """
f2 = cuadrados_minimos(x,y,2)
x_full = np.arange(0,100,0.5)
f_x_full = [f2(xi) for xi in x_full]

plt.figure()
plt.scatter(x,y,label="puntos")
plt.plot(x_full,f_x_full,label="cuadrado minimos")
plt.plot([0,100],[8,8],label="8 final")
plt.grid() 
plt.legend()


""" Ejercicios 4 """
x_full = np.linspace(-1,1,40)
y_full =  1 / (1 + 25*(x_full**2))

for n in [5,10,15]:   
    x = np.linspace(-1,1,n+1)
    y =  1 / (1 + 25*(x**2))
    
    f = newton_dif(x,y)
    y_int = np.array([f(xi) for xi in x_full])
    
    f1 = cuadrados_minimos(x,y,int(2*n/5))
    f2 = cuadrados_minimos(x,y,int(4*n/5))
    f_x1_full = [f1(xi) for xi in x_full]
    f_x2_full = [f2(xi) for xi in x_full]    
    
    plt.figure()
    plt.scatter(x,y)
    plt.plot(x_full,y_full,label="f")
    plt.plot(x_full,y_int,label="interpolado")
    plt.plot(x_full,f_x1_full,label="CM "+str(2*n/5))
    plt.plot(x_full,f_x2_full,label="CM "+str(4*n/5))
    plt.legend()

""" Ejercicios 5 """
r = 1 # radio
angulo = np.linspace(0,2*np.pi,64)
x = r*np.cos(angulo)
y = r*np.sin(angulo)
x_elipse = 4*x
y_elipse = 3*x + 5*y

A = np.array([[4,0],[3,5]])

A1 = 2*np.sqrt(10)*np.array([1/np.sqrt(5),2/np.sqrt(5)])
A2 = np.sqrt(10)*np.array([2/np.sqrt(5),-1/np.sqrt(5)])
""" Notamos que A*v1 = singular1*u1
                A*v2 = singular2*u2
"""
plt.figure()
plt.plot(x,y)
plt.plot(x_elipse,y_elipse)
plt.plot([0,1/np.sqrt(2)],[0,1/np.sqrt(2)], label="v1")
plt.plot([0,1/np.sqrt(2)],[0,-1/np.sqrt(2)], label="v2")
plt.plot([0,A1[0]],[0,A1[1]],label="A*v1")
plt.plot([0,A2[0]],[0,A2[1]],label="A*v2")
plt.grid()
plt.legend()
plt.gca().set_aspect('equal') # Para que no se vea como un ovalo



""" Ejercicios 8 """
def aproxima(x,y,S):
    n = len(S) # n columnas
    nx = len(x) # nx filas
    A = np.zeros((nx,n)) 
    for i in range(nx):
            for j in range(n):
                A[i][j] = S[j](x[i])
    
    A_ = np.dot(np.transpose(A),A)  # A.T si A no es una matriz, no puedo
    b = np.dot(np.transpose(A),y)
    coef = np.dot(npl.inv(A_) , b)
    print(coef)
    
    def function(m):
        X = np.ones(len(coef))
        for i in range(len(coef)):
            X[i] = coef[i]*(S[i](m))
        return sum(X)
    return(function)  


def dos_elevado(x): return 2**x
def tres_elevado(x): return 3**x
def uno(x): return 1

x = [0, 1, 2, 3]
y = [0.3, -0.2, 7.3, 23.3]
Sa = [dos_elevado,tres_elevado]
fa = aproxima(x,y,Sa)

Sb = [dos_elevado,tres_elevado,uno]
fb = aproxima(x,y,Sb)

x_full = np.linspace(0,3,32)
ya = [fa(xi) for xi in x_full]
yb = [fb(xi) for xi in x_full]

plt.figure()
plt.scatter(x,y,label="puntos")
plt.plot(x_full,ya,label="2**x+3**x")
plt.plot(x_full,yb,label="2**x+3**x+1")
plt.grid()
plt.legend()

""" Ejercicio 12 """
from scipy.special import erf

x_full = np.linspace(-15,15,60)
y_full = [erf(xi) for xi in x_full]
plt.figure()
plt.plot(x_full,y_full,label="erf")
plt.legend()

#b)
X = np.linspace(-10,10,20)
Y = [erf(xi) for xi in X]
plt.figure()
plt.plot(x_full,y_full,label="puntos")
for n in [1,3,5]:  
    f = cuadrados_minimos(X,Y,n)
    y_int = [f(xi) for xi in x_full]
    plt.plot(x_full,y_int,label="pol: "+str(n))   
plt.grid()
plt.legend()

#c)
def f1(x): return x*(np.e**(-x**2) )
def f2(x): return np.arctan(x)
def f3(x): return x/(x**2 + 1)

S = [f1,f2,f3]
f = aproxima(X,Y,S)
fx = [f(xi) for xi in x_full]

plt.figure()
plt.plot(x_full,y_full,label="erf")
plt.plot(x_full,fx,label="aproximacion")
plt.grid()
plt.legend()


""" Ejercicio 13 """
x = np.array([-1,0,1,2])
y = np.array([8.1,3,1.1,0.5])
ln_y = np.log(y)

f = cuadrados_minimos(x,ln_y,1) # LINEAL
a = 3.0528532038321656
b = -0.93583358
def eje13(x): return a*np.e**(b*x)

x_full = np.linspace(-1,2,32)
y_int = [eje13(xi) for xi in x_full]
plt.figure()
plt.scatter(x,y,label="puntos")
plt.plot(x_full,y_int,label="aproximacion")
plt.grid()
plt.legend()


""" Ejercicio 14 """
x = np.array([-1,0,1,2])
y = np.array([-1.1, -0.4, -0.9, -2.7])
ln_y = np.log(-y)

f = cuadrados_minimos(x,ln_y,2) # CUADRATICA
a = 0.5275533
b = -0.1770778  
c = -0.68606337
def eje14(x): return -np.e**(a*x**2 + b*x + c)

x_full = np.linspace(-1,2,32)
y_int = [eje14(xi) for xi in x_full]
plt.figure()
plt.scatter(x,y,label="puntos")
plt.plot(x_full,y_int,label="aproximacion")
plt.grid()
plt.legend()


""" Ejercicio 15 """
x = np.arange(-np.pi,np.pi,0.001)
y = np.sign(x)
def cos2(x): return np.cos(x*2)
def sen2(x): return np.sin(x*2)
def cos3(x): return np.cos(x*3)
def sen3(x): return np.sin(x*3)

S = [np.cos,np.sin,cos2,sen2,cos3,sen3]
f = aproxima(x,y,S)
y_int = [f(xi) for xi in x]

plt.figure()
plt.plot(x,y,label="puntos")
plt.plot(x,y_int,label="aproximacion")
plt.grid()
plt.legend()

