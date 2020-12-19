# En python, las matrices las construimos usando arrays
import numpy as np
M = np.array([[1,2],[3,4]]) # Un array es una lista con listas adentro. Cada lista representa una fila
print(M)

# La función dot nos permite hacer productos entre matrices. Un vector asimismo lo podemos pensar como una matriz

v = np.array([1,1])

print(np.dot(M,v))
# Podemos trasponer
print(np.transpose(M))
print(M.T)
""" 
    Podemos realizar operaciones de inversa
    Estas estan en el submodulo linalg de numpy
    Donde estan operaciones de Algebra Lineal
"""
print(np.linalg.inv(M))
# Para que sea más comodo:
import numpy.linalg as npl
print(np.dot(M,npl.inv(M)))
# Noten como el error de punto flotante aparece como error en la inversion

# Para calcular la norma de una matriz
# Suma los elementos al cuadrado
print(npl.norm(v))
print(npl.norm(M))

# se pueden calcular otras normas, checkeen: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html

npl.norm(M,ord=np.inf) # Norma infinito, el elemento más grande.
npl.norm(v,ord=1) # Norma 1, suma de los elementos. En matrices da suma de columnas
npl.norm(M,ord='fro') # Norma frobenius

# Podemos calcular determinantes

print(npl.det(M))

# Podemos calcular autovalores y autovectores

print(npl.eig(M)) # Nos devuelve una lista, primero con autovalores, luego con matriz de autovectores

# Y podemos calcular el número de condicion de una matriz tambien

print(npl.cond(M))

# Las funciones que encuentran máximos y mínimos funcionan sobre matrices
print(np.max(M))
print(np.argmax(M)) # Fijense que devuelve la posición como si pusieramos una fila al lado de la otra

# También tenemos algunas funciones por default

print(np.ones((10,10))) # Matriz llena de unos, de 10x10
print(np.zeros((10,10))) # Matriz llena de ceros, de 10x10
print(np.full((10,10),5)) # Matrix llena de cincos, de 10x10


# Obtener la diagonal de una matriz (agrego al final porque no salio en el video)
print(np.diagonal(M)) # La devuelve como array
# Obtener una matriz identidad 
print(np.identity(10)) # matriz identidad, el argumento da el tamaño de la diagonal
print(np.diag([10,2])) # matriz con diagonal dada por la lista / array y ceros por fuera
print(np.eye(10,2)) # 1s en la diagonal, 0s fuera, de N (primer argumento) x M (segundo argumento)
