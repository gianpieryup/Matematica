## EJERCICIO 5
## NOMBRE APELLIDO: Gianpier Yupanqui
## LU 819/18

import matplotlib.pyplot as plt
import numpy as np
import imageio

def matriz_evolucion(N,k,h):
    landa = k / h
    resultado = []
    for q in range(N):
        fila = []
        for r in range(N):
            if q==r: # Si estoy en la diagonal
                fila.append(2-2*landa**2)
            elif q==r-1 or q==r+1: # Si estoy en 1 menos que la diagonal o 1 mas
                fila.append(landa**2)
            else: # Si no es ningun caso
                fila.append(0)
        resultado.append(fila)
    return(np.array(resultado))

def f_x(x):# X es un array
    # f es la condicion inicia u(x,t=0)
    return np.sin(x) 

def g_x(x):
    # g es la condicion inicial du/dt(x,t=0)
    return x*0 

def cumple_contorno(solucion,contornos):
    """ contornos = [0,1] ejem:
        solucion = [1.2, 1.3, 1.4, ..., 1.8,2] , no cumple que el primero =0 y ultimo =0
        Mi solucion ya sea cumpla o no que los bordes son [0,1], la fuerzo
    """
    solucion[0] = contornos[0]
    solucion[-1] = contornos[1]
    """Los listas se pasa por defecto por referencia(Si se quita aca,quita en el paso_integracion (uN =) de  (cumple_contorno(...)))"""
    return solucion

def gen_condicion_inicial(N,f_x,g_x,k,h,contornos):
    """ Asumo  x0=0, x1= x0 +h,..., xN-1 =h*(N-1) """
    X_i = np.array([n*h for n in range(N)])
    condicion_0 = f_x(X_i)
    condicion_1 = f_x(X_i) + k*g_x(X_i)
    #Para asegurar nuestros extremos
    cumple_contorno(condicion_0,contornos)
    cumple_contorno(condicion_1,contornos)
    return([condicion_0,condicion_1])   

def paso_integracion(u,u_ ,contornos,A): 
    # U^j = u   y U^j-1 = u_ => uN = A*u + u_
    uN = np.dot(A,u) - u_
    uN = cumple_contorno(uN,contornos) #Verifica que la nueva Solucion tenga los bordes correctos
    return(uN) 

def integra(condicion_inicial,contornos,k,h,t0,T):
    """ condicion_inicial = [U^0,U^1]
    Mi intencion es hallar U^2, U^3 .. y guardarlos todo en una lista solucion"""
    solucion = condicion_inicial
    t = [t0]
    A = matriz_evolucion(len(condicion_inicial[0]),k,h)
    while t[-1]<T:
        tN = t[-1] + k
        uN = paso_integracion(solucion[-1],solucion[-2],contornos,A)
        t.append(tN)
        solucion.append(uN)
    return [t,solucion]


# H)    
# Ahora evaluamos condiciones iniciales y valores de contorno    
contornos = [0.0,0.0] # 0 en ambos extremos
N = 100 # Cantidad de pasos en la linea
# Notar la longitud es pi, dividido en segmentos de tamaña h
h = np.pi/(N-1) # Paso espacial, notar el N-1. 
""" Prueben sacar el -1 a ver que pasa
    RPTA: Como L=xN-1 =h*(N-1) el ultimo valor
    L= pi           con  h = np.pi/(N-1)
    L= (pi/N)*N-1   con  h = np.pi/N
    Por lo que el ultimo valor no es PI y la cuerda con longitud L != de pi
    En las graficas L no llegaria a ser PI, por lo tanto veremos en el caso inicial, el ultimo valor sen(!pi) != 0
    Como en las iteraciones se desplaza el error, vemos que la deformacion tambien se desplaza en toda la onda mientras oscila
"""
k = h**2/10 # Paso temporal
T = 8 # Tiempo final
t0 = 0 # tiempo inicial

# I)
# 1)
#INTEGRAMOS
condicion_inicial = gen_condicion_inicial(N,f_x,g_x,k,h,contornos) 
soluciones = integra(condicion_inicial,contornos,k,h,t0,T)
u = soluciones[1] # Acá debería tener todas las soluciones, un array por cada tiempo 
t = soluciones[0] # Acá tenemos los tiempos
 
""" GRAFICAMOS
plt.plot(np.arange(0,N*h,h),u[0]) # Miremos una solucion la primera
plt.savefig('condicion_inicial.png')
"""
# Primero armemos una función que arme una imagen
def grafica_t(soluciones, indice,k,h):  
    plt.clf()     # borramos lo que hubiera antes en la figura
    x = np.arange(len(soluciones[1][indice]))*h # Genero una tira de valores x
    u = soluciones[1][indice] # agarro la solución indicada
    t = indice*k # Calculo el tiempo
    # grafico la curva en ese tiempo
    plt.plot (x,u,label='t='+str(np.round(t,4)))
    plt.ylim(-1.5,1.5) # Para que se vea toda la soguita
    plt.legend()
    
# Y ahora armamos una función que nos haga un videito o un gif
def video_evolucion(soluciones, nombre_video,k,h):
    lista_fotos =[] # aca voy a ir guardando las fotos
    for i in range ( len(soluciones[0])):
        if i %500==0: # esto es para guardar 1 de cada 500 fotos y sea menos pesado el giff. Si se les corta el guardado de imagen en repl, prueben aumentar este numero
            grafica_t(soluciones, i, k, h)
            plt.savefig(nombre_video + '.png')
            lista_fotos.append(imageio.imread(nombre_video + '.png'))
            print (str(i) + ' de '+ str(len(soluciones[0])) + ' fotos guardadas')
    imageio.mimsave(nombre_video + '.gif', lista_fotos) # funcion que crea el video
    print('Gif Guardado')
    
video_evolucion(soluciones,'primer_armonico',k,h)


# 2)
def f_x2(x):
   return np.sin(2*x)

condicion_inicial = gen_condicion_inicial(N,f_x2,g_x,k,h,contornos) 
soluciones = integra(condicion_inicial,contornos,k,h,t0,T)
video_evolucion(soluciones,'segundo_armonico',k,h)

# 3)
def f_x3(x):# x es un array
   res = []
   for elem in x:
       if (np.pi/3 <= elem) and (elem <= 2*np.pi/3):
           res.append(1)
       else:      
           res.append(0)
   return res

condicion_inicial = gen_condicion_inicial(N,f_x3,g_x,k,h,contornos) 
soluciones = integra(condicion_inicial,contornos,k,h,t0,T)
video_evolucion(soluciones,'escalonada',k,h)


"""
def g_x4(x):# x es un array
   res = []
   for elem in x:
       if (np.pi/100 <= elem) and (elem <= 2*np.pi/100):
           res.append(1)
       else:
           res.append(0)
   return np.array(res) 
# g_x = [0...0] de arriba
condicion_inicial = gen_condicion_inicial(N,g_x,g_x4,k,h,contornos) 
soluciones = integra(condicion_inicial,contornos,k,h,t0,T)
video_evolucion(soluciones,'patadita',k,h)
"""