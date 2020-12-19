import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from interpolacion import newton_dif

"""Ejercicio 1"""
def coef_inde(x,y):
    A = []
    for xi in x:
        res =[]
        for l in range(len(x)):
            res.append(xi**l)
        A.append(res)
    
    return A

x = [-1,0,2,3]
y = [-1,3,11,27]
c1 = np.dot(npl.inv(coef_inde(x,y)),y)

x = [-1,0,1,2]
y = [-3,1,1,3]
c2 = np.dot(npl.inv(coef_inde(x,y)),y)

"""--------------  Ejercicio 4  -------------"""

""" LO que debemos notar es que la funcion interpolada oscila mucho
    Entre mas puntos , con lo cual no se aproxima a REALMENTE A LA FUN original
"""
x_full = np.linspace(-1,1,40)

for n in [5,10,15]:
    """ F(x) = 1/ 1 + 25*(x**2) """
    x = np.linspace(-1,1,n+1)
    y = 1/(1 + 25*(x**2))
    f = newton_dif(x,y)
    y_int = np.array([f(xi) for xi in x_full])
    
    plt.figure()
    plt.scatter(x,y,label="puntos")
    plt.plot(x_full,y_int,label="f interpolado")
    plt.legend()

    """ F(x) = abs(x) """
    #x = np.linspace(-1,1,n+1)
    y = abs(x)
    f = newton_dif(x,y)
    y_int = np.array([f(xi) for xi in x_full])
    
    plt.figure()
    plt.scatter(x,y,label="puntos")
    plt.plot(x_full,y_int,label="f interpolado")
    plt.legend()

    """ F(x) = sen(pi*x) Si se comporta como deberia, no importa n """
    #x = np.linspace(-1,1,n+1)
    y = np.sin(x*np.pi)
    f = newton_dif(x,y)
    y_int = np.array([f(xi) for xi in x_full])
    
    plt.figure()
    plt.scatter(x,y,label="puntos")
    plt.plot(x_full,y_int,label="f interpolado")
    plt.legend()


"""--------------  Ejercicio 7  -------------"""
x_full = np.linspace(1,10,50)

x = np.linspace(1,10,5)
y = np.log(1/x)
fun_int = newton_dif(x,y)
y_int = np.array([fun_int(xi) for xi in x_full])
plt.figure()
plt.scatter(x,y,label="puntos")
plt.plot(x_full,y_int,label="f interpolado")



"""--------------  Ejercicio 15  -------------"""
def W(n,x):# aumentar 1 al n
    v_xi = [-1 + 2*i/n for i in range(n)]
    res = 1
    for xi in v_xi:
        res *= x - xi
    return res

def Tchebychev(k,x):# aumentar 1 al n
    return np.cos(k * np.arccos(x))

for n in [5,10,15]:
    xw = np.linspace(-1,1,20)
    yw = np.array([W(n+1,x) for x in xw])
    yt = np.array([Tchebychev(n+1,x) for x in xw])
    plt.figure()
    plt.plot(xw,yw,label="W n+1")   
    plt.plot(xw,yt,label="Tchebychev")
    plt.legend()
#NOTAR: W_n+1 <= 1 siempre Ejercicio 8b) con a=-1,b=1    

"""--------------  Ejercicio 16  -------------  """
x_full = np.linspace(-1,1,50)
y_full = 1/(1 + 25*(x_full**2))
plt.figure()
plt.plot(x_full,y_full,label="original") 
plt.legend()
for n in [6,11,16]:
    x = np.array([np.cos((2*i + 1)*np.pi /(2*n)) for i in range(n)])  # Los ceros de Tchebychef: 
    y = 1/(1 + 25*(x**2))
    fun_int = newton_dif(x,y)
    y_int = np.array([fun_int(xi) for xi in x_full])
    plt.figure()
    plt.scatter(x,y,label="puntos")
    plt.plot(x_full,y_int,label="interpolado")
    plt.legend()


y_full = abs(x_full)
plt.figure()
plt.plot(x_full,y_full,label="original") 
plt.legend()
for n in [6,11,16]:
    x = np.array([np.cos((2*i + 1)*np.pi /(2*n)) for i in range(n)])
    y = abs(x)
    fun_int = newton_dif(x,y)
    y_int = np.array([fun_int(xi) for xi in x_full])
    
    plt.figure()
    plt.scatter(x,y,label="puntos")
    plt.plot(x_full,y_int,label="interpolado")
    plt.legend()
    

y_full = np.sin(x_full*np.pi)
plt.figure()
plt.plot(x_full,y_full,label="original") 
plt.legend()
for n in [6,11,16]:
    x = np.array([np.cos((2*i + 1)*np.pi /(2*n)) for i in range(n)])
    y = np.sin(x*np.pi)    
    fun_int = newton_dif(x,y)
    y_int = np.array([fun_int(xi) for xi in x_full])
    
    plt.figure()
    plt.scatter(x,y,label="puntos")
    plt.plot(x_full,y_int,label="interpolado")
    plt.legend()   
    
    
    
""" Ejercicio 25 SPLIT CUBICO"""   
x_full = np.linspace(0,1,50)    
y_full = np.sin(np.pi*x_full)    
plt.figure()
plt.plot(x_full,y_full,label="original")
 
def S0(x): return -4*(x**3) + 3*x    
def S1(x): return  4*(x**3) - 12*(x**2) + 9*x -1

x1 = np.linspace(0,0.5,15) 
y1 = S0(x1)

x2 = np.linspace(0.5,1,15) 
y2 = S1(x2)

plt.scatter(x1,y1,label="SP1")
plt.scatter(x2,y2,label="SP2")
plt.legend()