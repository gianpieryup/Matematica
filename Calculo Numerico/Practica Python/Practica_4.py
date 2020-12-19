import numpy as np
import numpy.linalg as npl
import random as rd
import matplotlib.pyplot as plt

"""   Ejercicio 3  """
A = np.array([[2,0,5],[1,2,0],[0,4,1]])
x = [rd.random(),rd.random(),rd.random()] # el vector de 3x3
s = [0] 
for i in range(100):
    S_k = s[i]
    der = npl.norm(np.dot(A,x)) / npl.norm(x)
    S_k1 = max(S_k,der)
    s.append(S_k1)
    x = [rd.random(),rd.random(),rd.random()]

cost = npl.norm(A)*np.ones(100)
plt.plot(s,label='Aproximaciones')
plt.plot(cost,label='normaA')
plt.legend()

#  Ejercicio 6  
A = np.array([[3,0,0],[0,5/4,3/4],[0,3/4,5/4]])
b = np.array([3,2,2])
b_prima = np.array([3.0000009,2.0000009,2.000008])
x = [1,1,1]
if(npl.norm(b - b_prima) < 10**-4/12 ):
    print("Se cumple la cota de b")

x_prima = np.dot(npl.inv(A),b_prima)    
    
if(npl.norm(x - x_prima) < 10**-4 ):
    print("Se cumple la cota de x")