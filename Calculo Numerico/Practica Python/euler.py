def euler(f,t0,T,y0,h):
    """ Metodo de un paso (EULER)
        f es y'=f(t,x)
        t0 el valor inicial
        y0 = y(t0)
        T el valor a aproximar
        h el paso
    """
    t = [t0]
    y = [y0]
    N = int((T-t0)/h)
    for c in range(N):#son N operaciones
        yN = y[-1] + h*(f(t[-1],y[-1]))
        tN = t[-1] + h
        t.append(tN)
        y.append(yN)
    return [t,y]