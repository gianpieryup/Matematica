# Importamos numpy y matplotlib
import numpy as np
import matplotlib.pyplot as plt
# Importamos time, es un módulo que ya viene con Python, así que no hay que instalarlo
import time
# Esta función aplica el esquema que vimos en clase:
# t0: es el tiempo inicial
# tf: es el tiempo final
# x0: inicio del intervalo de posiciones
# xf: final del intervalo de posiciones
# a: constante del problema (para que funcione bien, debe valer que a > 0)
# g : función de valor de contorno u(0,t) = g(t)
# f : función de valor de contorno u(x, 0) = f(x)
# N: cantidad de puntos en los que queremos discretizar [x0, xf]
# M: cantidad de puntos en los que queremos discretizar [t0, tf]
def upwind(t0, tf, x0, xf, a, g, f, N, M):
    # Discretizamos [x0, xf] y calculamos h
    x = np.linspace(x0, xf, N)
    h = (xf - x0) / N
    # Discretizamos [t0, tf] y calculamos k
    t = np.linspace(t0, tf, M)
    k = (tf - t0) / M
    # Calculamos nu
    nu = k / h
    # Escribimos la matriz A con 1-a*nu en la diagonal y a*nu en la subdiagonal
    A = np.zeros((N, N))
    np.fill_diagonal(A, 1 - a * nu)
    np.fill_diagonal(A[1:N, :N - 1], a * nu)
    # Inicializamos una matriz que guarda las aproximaciones numericas. Esta matriz
    # tiene M filas y N columnas: en el lugar (j,i) tendremos la aproximación
    # numérica de u(x_i, t_j)
    sol_numerica = np.empty((M, N))
    # Escribimos u^0: en la coordenada i, este vector tiene el valor de f(x_i) = u(x,0) .
    # Lo agregamos como primera fila de la matriz de aproximaciones:
    u = np.array([f(X) for X in x])
    sol_numerica[0, :] = u
    # Realizamos las iteraciones del modelo
    for j in range(1, M):
        # Calculamos el vector g^j (recordar que t_j = j*k + t0)
        G = np.array([g(j * k + t0)] + [0] * (N - 1))
        # Calculamos los valores del siguiente paso temporal y los guardamos en la variable u
        u = A @ u + G
        # Agregamos el resultado de esta iteracion en la matriz de aproximaciones numericas
        sol_numerica[j, :] = u
        # Devolvemos la matriz de resultados y las discretizacion en x
    return sol_numerica, x




def ecuacion_de_transporte():
    # Definimos la funcion f(x) (corresponde al valor de contorno u(x,0) = f(x))
    def f(x):
        return np.exp(-10 * ((4 * x - 1) ** 2))
    # Definimos la funcion g(t) (corresponde al valor de contrno u(0, t) = g(t))
    def g(t):
        return 0
    # Definimos la solucion exacta de la ecuacion
    def u(x, t):
        return f(x - a * t)
    # Introducimos los datos de nuestro problema
    x0 = 0 # Inicio del intervalo espacial
    xf = 2 # Fin del intervalo espacial
    t0 = 0 # Tiempo inicial
    tf = 1 # Tiempo final
    # Estos valores cumplen con la condicion que garantiza estabilidad y convergencia
    N = 200
    M = 100
    # Introducimos el valor de la constante de la ecuacion y calculamos h y k
    a = 1
    h = (xf - x0) / N
    k = (tf - t0) / M
    # Utilizamos el método para obtener la matriz con las aproximaciones
    # numéricas y la discretizacion en x
    sol_numerica, xd = upwind(t0, tf, x0, xf, a, g, f, N, M)
    # Creamos un matriz donde guardamos el valor exacto de u(x_i, t_j) en su fila j y columna i
    sol_exacta = np.empty((M, N))
    for j in range(M):
        for i in range(N):
            sol_exacta[j, i] = u(h * i + x0, j * k + t0)
    
    """Aqui comienza la parte en la que elaboramos el gráfico animado"""
    # Abrimos una ventana sin ninguna figura o gráfico
    plt.show()
    # Creamos una nueva figura y pedimos que sólo nos muestre los
    # valores del eje x en [0, 1.5] y los valores del eje y en [0,2]
    axes = plt.gca()
    axes.set_xlim(0, 1.5)
    axes.set_ylim(0, 2)
    # Agregamos a la figura el gráfico de la aproximacion numerica a
    # tiempo 0, que es la primera fila de la matriz "sol_numerica"
    grafico_solnum, = axes.plot(xd, sol_numerica[0, :], 'r')
    # Agregamos a la figura el gráfico de la solucion exacta a tiempo 0,
    # que es la primera fila de la matriz "sol_exacta"
    grafico_solex, = axes.plot(xd, sol_exacta[0, :], 'b')
    
    # Ahora elaboraremos los "cuadros" de la animación,es decir, vamos a graficar para cada t_j
    for j in range(M):
    # Actualizamos el gráfico de la solucion numerica
    # En el eje x mantenemos los mismo datos
        grafico_solnum.set_xdata(xd)
        # Actualizamos los datos del eje y: usamos la fila j de sol_num, que corresponde a t_j
        grafico_solnum.set_ydata(sol_numerica[j, :])
        # Actualizamos el gráfico de la solucion exacta de manera análoga
        grafico_solex.set_xdata(xd)
        grafico_solex.set_ydata(sol_exacta[j, :])
        # Pedimos que dibuje la figura y los gráficos en la ventana que abrimos al principio
        plt.draw()
        # Pedimos que realice una breve pausa entre este cuadro y el siguiente
        plt.pause(1e-17)
        time.sleep(0.1)
        # Incluimos esto para que no se cierre la ventana con la animación cuando termine
        plt.show()
ecuacion_de_transporte()