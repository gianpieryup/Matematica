# Resolucióm, ecuación de calor

# Tenemos 

## u_N_i = (1-2*k/h**2)*u_i + k /h**2*(u_ip+u_im)

# donde  la N indica la actualización en el tiempo, 
# ip representa i+1 y im representa im

# Podemos representa esto usando una matriz.
# Armemos una funcion que la construya!
import numpy as np
def matriz_evolucion(N,k,h):
    # Vamos a representarla armando una lista con listas.
    # La lista grande se mueve sobre las filas, y las internas sobre columnas
    resultado = []
    for q in range(N):
        fila = []
        for r in range(N):
            if q==r: # Si estoy en la diagonal
                fila.append(1-2*k/h**2)
            elif q==r-1 or q==r+1: # Si estoy en 1 menos que la diagonal o 1 mas
                fila.append(k/h**2)
            else: # Si no es ningun caso
                fila.append(0)
        resultado.append(fila)
    return(np.array(resultado))

# Un ejemplito:
A = matriz_evolucion(4,0.1**3,0.1)        
print(A)

# Vamos a armar una función que genere una condicion inicial
# Deltiforme (un pincho)

def genera_deltiforme(tamaño,posicion,piso,pico):
    condicion_inicial = np.repeat(piso,tamaño)
    condicion_inicial[posicion] = pico
    return(condicion_inicial)
    
# Fijense que algo clave es que tenemos que revisar
# que se cumpla la condicion de contorno, y si no se cumple, forzarla
def cumple_contorno(solucion,contornos):
    solucion[0] = contornos[0] # Noten que como solucion es una lista
    solucion[-1] = contornos[1] # En realidad no haría falta que la retornemos.
    return(solucion)
    
g = genera_deltiforme(10,3,0,10) 
print(g)

cumple_contorno(g,[0,0])

# Listo! Con esto podemos armar la evolución

def paso_integracion(u,contornos,matriz): # Esta vez, ya vamos a tomar como construida la matriz
    uN = np.dot(matriz,u)
    uN = cumple_contorno(uN,contornos)
    return(uN)

def integra(condicion_inicial,contornos,k,h,t0,T):
    solucion = [condicion_inicial]
    t = [t0]
    A = matriz_evolucion(len(condicion_inicial),k,h)
    while t[-1]<T:
        tN = t[-1] + k
        uN = paso_integracion(solucion[-1],contornos,A)
        t.append(tN)
        solucion.append(uN)
    return([t,solucion])
    
    
# Probemos ahora algo facil de entender: una linea
# con condiciones iniciales 1 y 0:
    
h = 0.01
k = h**2/10
contornos = [1.0,0.0]
L = 100 # Tamaño de la linea
T = .4 # Tiempo final
t0 = 0 # tiempo inicial
condicion_inicial = cumple_contorno(genera_deltiforme(L,0,0.0,0.0),contornos) # Así es una condicion inicial chata
soluciones = integra(condicion_inicial,contornos,k,h,t0,T)
t = soluciones[0]
u = soluciones[1]
# Grafiquemos alguna solucion intermedia
import matplotlib.pyplot as plt
plt.plot(np.arange(0,L*h,h),u[0])
plt.savefig('solucion_intermedia.png')

# Para armar un giff tenemos que hacer muchas imagenes, y guardarlas
# Primero armemos una función que arme una imagen
def grafica_t(soluciones, indice,k,h):  
    # borramos lo que hubiera antes en la figura
    plt.clf()
    x = np.arange(len(soluciones[1][indice]))*h
    u = soluciones[1][indice]
    t = indice*k
    # grafico la curva en ese tiempo
    plt.plot (x,u,label='t='+str(np.round(t,4)))
    plt.legend()
# Y ahora armamos una función que nos haga un videito o un gif
import imageio
def video_evolucion(soluciones, nombre_video,k,h):
    lista_fotos =[] # aca voy a ir guardando las fotos
    for i in range ( len(soluciones[0])):
        if i %500==0: # esto es para guardar 1 de cada 2 fotos y tarde menos
            grafica_t(soluciones, i, k, h)
            plt.savefig(nombre_video + '.png')
            lista_fotos.append(imageio.imread(nombre_video + '.png'))
            print (str(i) + ' de '+ str(len(soluciones[0])) + ' fotos guardadas')
    imageio.mimsave(nombre_video + '.gif', lista_fotos) # funcion que crea el video
    print('Video Guardado')
    
video_evolucion(soluciones,'calor',k,h)