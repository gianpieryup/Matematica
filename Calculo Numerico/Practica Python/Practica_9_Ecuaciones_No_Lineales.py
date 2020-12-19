import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

#1)
def bizexion(f, a, b, tol):
    res = []
    while b - a >tol:
        c = 0.5 * (a + b)
        if f(c) >= 0:
            b = c
        else:
            a = c
        res.append(c)
    return res

#2)
def ej2(m): return 2*m - np.tan(m)
res = bizexion(ej2, -1, 1, 10**-5)

x_full = np.linspace(-1,1,100)
y_int = ej2(x_full)


plt.title("2*x - tan(x)")
plt.plot(x_full,y_int,label="funcion")
plt.grid()
plt.show()


#3)
def newton_raphson(f,f_prima,x0,error):
    xn = x0
    res = []
    while abs(f(xn)) > error:
        xn = xn - f(xn)/f_prima(xn)
        res.append(xn)     
    return res

#4)
def secante(f,x0,x1,error):
    a = x0
    b = x1
    res = []
    c = b - f(b)*(b-a)/(f(b)-f(a))
    while abs(f(c)) > error:
        c = b - f(b)*(b-a)/(f(b)-f(a))
        res.append(c)
        a = b
        b = c
    return res

#5)
def raiz_cubica(x): return x**3 - 2
def r_prima(x): return 3*x**2

res_bis = np.array(bizexion(raiz_cubica, 1, 2,10**-6))
res_NR = np.array(newton_raphson(raiz_cubica,r_prima,2,10**-6))
res_sec = np.array(secante(raiz_cubica,3,2,10**-6))   

# 2**(1/3) =  1.2599210498948732
error_bis = np.log(np.abs(2**(1/3) - res_bis))
error_NR = np.log(np.abs(2**(1/3) - res_NR))
error_sec = np.log(np.abs(2**(1/3) - res_sec))

plt.title("Orden convergencia")
plt.scatter(np.arange(len(error_bis)),error_bis,label="biseccion")
plt.scatter(np.arange(len(error_NR)),error_NR,label="Newtom-Rapshon")
plt.scatter(np.arange(len(error_sec)),error_sec,label="Secante")
plt.xlabel("Iteracion")
plt.ylabel("log(e_n)")
plt.grid()
plt.show()



#16)
def newton_raphson(f,f_prima,x0,max_iter,tol):
    xn = x0
    res = []
    iter = 0
    while abs(f(xn)) > tol and max_iter > iter:
        xn = xn - f(xn)/f_prima(xn)
        res.append(xn)  
        iter+=1
    return res


def newton_raphson_2(f,f_prima,x0,max_iter,tol):
    xn = x0
    res = []
    iter = 0
    while abs(f(xn)) > tol and max_iter > iter:
        xn = xn - 2*f(xn)/f_prima(xn)
        res.append(xn)  
        iter+=1
    return res

def fun_16(x): return 4*x**3 - 3*x + 1
def f_prima(x): return 12*x**2 -3

x0 = 25
res = newton_raphson(fun_16,f_prima,x0,10,10**-5)
res2 = newton_raphson_2(fun_16,f_prima,x0,10,10**-5)

plt.title("Diferencias")
plt.plot(np.arange(len(res)),res,label="NR")
plt.plot(np.arange(len(res2)),res2,label="NR2")
plt.grid()
plt.show()


#17)
x0 = np.array([2,1])
def f(x,y):
    return [y*np.e**(-x**2)*(-2*x) + 2*x + 1, np.e**(-x**2) + 4*y**3]

def Df(x,y):
    return [[y*np.e**(-x**2)*(4*x**2 - 2) + 2, -2*x*np.e**(-x**2)],
             [np.e**(-x**2)*-2*x, 12*y**2]]

def newton_raphson_generalizado(f,Df,x0,max_iter):
    xn = x0
    res = []
    iter = 0
    while max_iter > iter:
        xn = xn - np.dot(npl.inv(Df(xn[0],xn[1])), f(xn[0],xn[1]))
        res.append(xn)  
        iter+=1
    return res

res = newton_raphson_generalizado(f,Df,x0,10)
print("_______________________________________________")
print("             x         |             y")
for xi in res:
    print('{0:22} | {1}'.format(xi[0], xi[1]))
    
# 18)
from euler import euler

def y_prima(t,x): return x**2 - x**3
lamba = 0.01
h = 2
t0 = 0
y0 = lamba
T = 2/lamba
res = euler(y_prima,t0,T,y0,h)
 
plt.title("Ejericio 18")
plt.plot(res[0],res[1],label="euler expl")  

#-----------------
# h=2
def fun_int(x,yn): return h*x**3 - h*x**2 + x - yn
def dev_fun_int(x): return 3*h*x**2 - 2*h*x + 1

def iteracion(yn,error):# yn es y0
    y0 = yn # el y0 es coeficiente de la fun_int [lo usare en cada iteracion]
    while abs(fun_int(yn,y0)) > error:
        yn = yn - fun_int(yn,y0)/dev_fun_int(yn)
    return yn

t = [t0]
y = [y0]
N = int((T-t0)/h)
for c in range(N):#son N operaciones
    yN = iteracion(y[-1],10**-5)
    tN = t[-1] + h
    t.append(tN)
    y.append(yN)
    
plt.plot(t,y,label="euler impl")
plt.grid() # La grilla
plt.legend() # Los label
plt.show() # Que aca termina la grafica
