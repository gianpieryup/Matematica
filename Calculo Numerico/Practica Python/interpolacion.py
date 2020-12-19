import numpy as np
import numpy.linalg as npl

def pol_i(x,i,m):
    res = 1
    j=0
    while j < i:
        res *= m - x[j]
        j+=1
    return res

def matriz_dif(x):
    A = np.zeros((len(x),len(x)))
    A[:,0] = 1
    for q in range(len(x)):
        for r in range(1,q+1):
            A[q,r] = pol_i(x,r,x[q])  
    return(A)
    
def newton_dif(x,y):
    A = matriz_dif(x)
    C = np.dot(npl.inv(A),y)
    def function(m):
        X = np.ones(len(C))
        for i in range(len(C)):
            X[i] = C[i]*pol_i(x,i,m)
        return sum(X)
    return(function)
