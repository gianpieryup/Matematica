import numpy as np
import numpy.linalg as npl
def householderQR(A):
    # Lo copiamos tal cual de la práctica
    m = A.shape[0] # Filas
    n = A.shape[1] # Columnas
    R = np.copy(A) # Hacemos esto así para evitar que nos cambie A al operar
    Q = np.identity(m) # Lo empezamos como identidad para multiplicarlo luego
    for i in range(n):
        vk = np.copy(R[:,i]) #Toda la columna i
        q = 0
        while q<i: # Acá pongo ceros hasta justo antes de i
            vk[q] = 0
            q = q+1

        vk[i] = vk[i] + npl.norm(vk)*np.sign(vk[q]) # En i aplico el método
        vk = np.transpose(np.array([vk])) # TRANSFORMO a un array 
        Hk = np.identity(m) - 2*vk*np.transpose(vk)/npl.norm(vk)**2 
        R = np.dot(Hk,R) # Actualizo valores de R y de Q
        Q = np.dot(Hk,Q) 
    Q = npl.inv(Q) # Ah, pero Q era la inversa
    return(Q,R) # Retorno!
        
### Lo aplicamos al ejemplo de la práctica    
A = np.array([[1,-1,-1],[2,0,1],[-2,7,1]])
Q,R = householderQR(A)
print(Q)
print(R)

""" Comparar con la funcion (QR) de numpy/linalg"""
Qpy,Rpy = npl.qr(A)
print(Qpy)
print(Rpy)