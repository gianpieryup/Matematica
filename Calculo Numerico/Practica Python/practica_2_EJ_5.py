import numpy as np
import matplotlib.pyplot as plt
from taylor_orden_2 import taylor
from euler import euler

def tao_taylor(t,y,h):
    return y*(1+h/2)
 
def tao_euler(t,y):#podemos mejorar y tener na funcion llamada metodo_un_paso
    return y

list_h = [0.1,0.0625,0.05,0.025,0.01]
t0 = 0
y0 = 1
T = 1
list_eh_euler = []
list_eh_taylor = []
for h in list_h:
    vec = euler(tao_euler,t0,T,y0,h)
    e_h = abs(np.e - vec[1][-1])#Que estupido pense que y(1) =1 pero no, es 'e'
    list_eh_euler.append(e_h)
    print(e_h)
    vec = taylor(tao_taylor,t0,T,y0,h)
    e_h = abs(np.e - vec[1][-1])#Iigual
    list_eh_taylor.append(e_h)
""" Si lo vemos de esta forma
plt.plot(list_h,list_eh_euler,label = 'euler')
plt.plot(list_h,list_eh_taylor,label = 'taylor°2')
plt.grid()
plt.legend()
    Notaremos a medida que h (sea mas chiquito) el error disminuye, pero como comparamos
    Estos dos metodos cuando h es mas chiquito, si se pegan tanto?
"""    
plt.plot(np.log(list_h),np.log(list_eh_euler),label = 'euler')
plt.plot(np.log(list_h),np.log(list_eh_taylor),label = 'taylor°2')
plt.xlabel('log(h)')
plt.ylabel('log(e_h)')
plt.grid()
plt.legend()

""" Recordar que el logaritmo es creciente por lo tanto
    log(0.1) > log(0.0625) > ...
    Al meter LOGARITMO podemos ver con mayor comodidad el comportamiento de las curvas , pero relacionadas entre ellas
    Veremos que son dos rectas paralelas
    EULER: e_h ~ h => log(e_h) ~ log(h)
    TAYLOR: e_h ~ h^2 => log(e_h) ~ log(h^2)=2*log(h)    
    Mas alla de la exactitud de los valores dado que h no es muy chiquito
    La idea es ver que la proporcion entre los metodos es paralela
"""