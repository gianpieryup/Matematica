import numpy.linalg as npl
import numpy as np
import matplotlib.pyplot as plt

# A
A = np.array([[1,2,3],[4,5,6]])
U,s,V = npl.svd(A)
# A.shape = (2,3)  // 2 : #filas y 3 : #columnas
# s = [g1,g2,..]  // los elementos de la diagonal, no la matriz diagonal
S = np.zeros((A.shape[0], A.shape[1]))
S[:len(s), :len(s)] = np.diag(s)
A_ = np.dot(U,np.dot(S,V))
print(np.mean(A_-A)) # Esto es por el error de los flotantes, pero es pequeño


# B
import imageio as img
image = img.imread('arbol.jpg',format='jpg')

def a_grises(image,r,g):
    R = image[:,:,0] # Matriz de rojos
    G = image[:,:,1] # Matriz de grises
    B = image[:,:,2] # Matriz de azules
    gris = r*R + g*G + (1-r-g)*B
    return(gris)
    
r=g=1/3  # Este es mas oscuro
imagen_gris = a_grises(image,r,g)
plt.imshow(imagen_gris,cmap='gray',vmin=0,vmax=255)
plt.savefig('ejemplo_en_escala_de_grises_1.jpg')

r=0.3
g=0.59   # Este es mas claro
imagen_gris = a_grises(image,r,g)
plt.imshow(imagen_gris,cmap='gray',vmin=0,vmax=255)
plt.savefig('ejemplo_en_escala_de_grises_2.jpg')
    
# C
A = imagen_gris
U,s,V = npl.svd(A)
S = np.zeros((A.shape[0], A.shape[1]))
S[:len(s), :len(s)] = np.diag(s)
A_ = np.dot(U,np.dot(S,V))    
plt.imshow(A_,cmap='gray',vmin=0,vmax=255)
plt.savefig('escala_de_grises_SVD.jpg')
# Si es igual, son muy identicos

# D
def reduce_svd(A,p):
    U,s,V = npl.svd(A)
    n_elementos = int(p*len(s))
    s[len(s)-n_elementos:] = 0 #Esto funciona si es np.array no si es solo una lista
    S = np.zeros((A.shape[0], A.shape[1]))
    S[:len(s), :len(s)] = np.diag(s)
    A_ = np.dot(U,np.dot(S,V))
    return(A_)

# En las iteraciones esta el 90%
# E
image = imagen_gris
error = []
for p in [0.9,0.8,0.5,0.1]:
    print('Calculando con p=' + str(p))
    image_ = reduce_svd(image,p)
    plt.imshow(image_,cmap='gray',vmin=0,vmax=255)
    plt.savefig('SVD_reducido_' +str(p*100)+'p.jpg')
    error.append(np.mean(np.abs(image_-image))/np.mean(image))

""" Cuantos autovalores considera necesarios para aproximar la imagen?
    AL menos el 20% de, si es menor la grafica se ve muy borrosa
    Es mas el fondo blanco se vuelve gris en algunos puntos
    Es aun mas, para mi con que el error sea < 0.01 seria lo mas optimo
    
"""

# F
from tqdm import tqdm  # Te muestra la barra de progreso,Nro Iteracion, Tiempo de ejecucion  AMAISING
#from time import sleep # Para agregar delay dentro del for//  sleep(0.01)
array_p = np.arange(0,1,.05) # ESTO LO PUSO EL PROFE

# SE PUEDE MEJORAR CON UN FOR DENTRO DE UN FOR  
# imagenes = ['arbol.jpg','mona_lisa.jpg','fractal.jpg','poligono.jpeg','cuadrado.jpg']

print(" Fractal ")
image = img.imread('fractal.jpg',format='jpg')
image = a_grises(image,r,g)
error_fractal = []    
for p in tqdm(array_p):
    image_ = reduce_svd(image,p)
    error_fractal.append(np.mean(np.abs(image_-image))/np.mean(image))


print(" Mona Lisa ")
image = img.imread('mona_lisa.jpg',format='jpg')
image = a_grises(image,r,g)
error_mona = []    
for p in tqdm(array_p):
    image_ = reduce_svd(image,p)
    error_mona.append(np.mean(np.abs(image_-image))/np.mean(image))


print(" Cuadrado  ")
image = img.imread('cuadrado.jpg',format='jpg')
image = a_grises(image,r,g)
error_cuadrado = []    
for p in tqdm(array_p):
    image_ = reduce_svd(image,p)
    error_cuadrado.append(np.mean(np.abs(image_-image))/np.mean(image))


print(" Poligono  ")
image = img.imread('poligono.jpeg',format='jpeg')
image = a_grises(image,r,g)
error_poligono = []    
for p in tqdm(array_p):
    image_ = reduce_svd(image,p)
    error_poligono.append(np.mean(np.abs(image_-image))/np.mean(image))

plt.figure()
plt.plot(array_p,error_fractal,label="error_fractal")
plt.plot(array_p,error_mona,label="error_mona")
plt.plot(array_p,error_cuadrado,label="error_cuadrada")
plt.plot(array_p,error_poligono,label="error_polig")
plt.xlabel('-% autovalores')
plt.ylabel('Error')
plt.legend()