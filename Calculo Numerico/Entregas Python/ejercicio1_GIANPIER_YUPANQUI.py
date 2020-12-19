## EJERCICIO 1
## NOMBRE APELLIDO: Gianpier Yupanqui
## LU 819/18


# La ecuacion de la recta es y=m*x+b
y1 = 10
y2 = 100
x1 = 10
x2 = 25

m = (y2-y1)/(x2-x1)
b = y1-m*x1
print('El valor de m es',m)
print('El valor de b es',b)


"""Probar reemplazar los valores de x1,y1,x2,y2 por otros."""
y1 = 5
y2 = 4
x1 = 5
x2 = 4
m = (y2-y1)/(x2-x1)
b = y1-m*x1
print('El valor de m es',m)
print('El valor de b es',b)



# La ecuacion de la recta es y=b*e**(m*x)
y1 = 10
y2 = 100
x1 = 10
x2 = 25
import numpy as np
m = np.log(y2/y1) / (x2 -x1)
b = y1 /(np.e **(m*x1))
print('El valor de m es',m)
print('El valor de b es',b)