import matplotlib.pyplot as plt
import numpy as np

def matriz_evolucion(N,k,h):
    # Vamos a representarla armando una lista con listas.
    # La lista grande se mueve sobre las filas, y las internas sobre columnas
    resultado = []
    for q in range(N):
        fila = []
        for r in range(N):
            if q==r: # Si estoy en la diagonal
                fila.append(2*(1-(k/h)**2))
            elif q==r-1 or q==r+1: # Si estoy en 1 menos que la diagonal o 1 mas
                fila.append((k/h)**2)
            else: # Si no es ningun caso
                fila.append(0)
        resultado.append(fila)
    return(np.array(resultado))


def f_x(x):
   return(np.sin(x))

def g_x(x):
    g = x
    return(0*g)

def cumple_contorno(solucion,contornos):
    solucion[0] = contornos[0] # Noten que como solucion es una lista
    solucion[-1] = contornos[1]
    return(solucion)

def gen_condicion_inicial(N,f_x,g_x,k,h,contornos):
    x = np.arange(0,N)*h
    condicion_0 = f_x(x)
    condicion_1 = f_x(x) + k * g_x(x)
    condicion_0 = cumple_contorno(condicion_0,contornos)
    condicion_1 = cumple_contorno(condicion_1,contornos)
    return([condicion_0,condicion_1])
    

def paso_integracion(u,u_,contornos,A): # Esta vez, ya vamos a tomar como construida la matriz
    uN = np.dot(A,u)-u_
    uN = cumple_contorno(uN,contornos)
    return(uN)

def integra(condicion_inicial,contornos,k,h,t0,T):
    solucion = condicion_inicial
    t = [t0]
    A = matriz_evolucion(len(condicion_inicial[0]),k,h)
    while t[-1]<T:
        tN = t[-1] + k
        u = solucion[-1]
        u_ = solucion[-2]
        uN = paso_integracion(u,u_,contornos,A)
        t.append(tN)
        solucion.append(uN)
    return([t,solucion])
    
    
# Probemos ahora algo facil de entender: una linea
# con condiciones iniciales 1 y 0:
    
contornos = [0,0.0]
N = 100 # Cantidad de pasos en la linea
h = np.pi/(N-1) # Paso espacial
k = h**2/10 # Paso temporal
T = 8 # Tiempo final
t0 = 0 # tiempo inicial
condicion_inicial = gen_condicion_inicial(N,f_x,g_x,k,h,contornos) 
soluciones = integra(condicion_inicial,contornos,k,h,t0,T)
soluciones[1][0]
u = soluciones[1]
t = soluciones[0]


### Para graficar, vamos a recortar un poco los valores, para evitar sobrecargar el gráfico
paso=500
t_grid = t[::paso]
x_grid = np.arange(0,N)*h
# Con este comando, creamos dos matrices, que sencillamente son una repetición de len(x_grid) * len(t_grid) de las tiras de datos x_grid y t_grid
x_grid,t_grid = np.meshgrid(x_grid,t_grid)
# u tiene que tener la forma de un array con una dimension con los elementos de x y la otra con los elementos de t. Por suerte ya lo tenemos así!
## Solo tenemos que recortarlo
u_grid = np.array(u[::paso])



# Modelo de curvas de nivel: En este formato la gráfica de parece a una serie de curvas de nivel
fig = plt.figure()
ax = plt.axes(projection='3d')
densidad_lineas = 30 # Con este parámetro comandamos la densidad de lineas del ploteo
ax.contour3D(x_grid, t_grid, u_grid, densidad_lineas, cmap='binary')

# Podemos rotar la imagen
## Theta y phi son los ángulos de esféricas: theta cambia la elevación y phi gira alrededor del eje z (nota, theta va entre -pi/2 y pi/2)
# Prueben estos valores e interpreten
theta = 0
phi = 0
ax.view_init(theta, phi)
plt.savefig('theta0_phi0.png')
fig # Con esto, si están en python, pueden verlo en consola al plot. Sino al png!

theta = 0
phi = 90
ax.view_init(theta, phi)
plt.savefig('theta0_phi90.png')
fig

theta = 90
phi = 0
ax.view_init(theta, phi)
plt.savefig('theta90_phi0.png')
fig


# Algunos otros plots que son lindos
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x_grid, t_grid, u_grid, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
plt.savefig('surface.png')
fig


plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(x_grid, t_grid, u_grid, color='black')
plt.savefig('wireframe.png')
fig


# Para explorar: si grafican varios a la vez, al igual que con un plot comun y silvestre, se grafican uno encima del otro, y le pueden dar texturas combinandolos