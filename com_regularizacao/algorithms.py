import numpy as np

def soft_thresholding( c, step, gamma ):
    return np.sign( c ) * np.maximum( np.abs( c ) - step * gamma, 0 ) # numpy faz vetorização

def prox_wavelet( H, HT, x, step, gamma ):
    c = H( x )
    S = soft_thresholding( c, step, gamma )
    return HT( S )

def prox_tv(  ):
    pass

def ista_fixo( f, x0, grad_f, prox, step, gamma, H, HT, max_iter = 1000, tol = 1.0e-1  ):
    x = x0.copy()
    g_f = grad_f( x )
    iterations = 0
    iters = [ x.copy() ]
    while f( x ) > tol and iterations < max_iter:
        y = x - step * g_f
        x = prox( H = H, HT = HT, x = y, step = step, gamma = gamma )
        g_f = grad_f( x )
        iterations += 1
        iters.append( x.copy() )

    return iters

def ista_exato( f, x0, grad_f, prox, g, step, H, max_iter = 1000, tol = 1.0e-1  ):
    pass

def ista_backtracking( f, x0, grad_f, prox, g, step, H, max_iter = 1000, tol = 1.0e-1  ):
    pass

def fista_fixo( f, x0, grad_f, prox, g, step, H, max_iter = 1000, tol = 1.0e-1  ):
    pass

def fista_exato( f, x0, grad_f, prox, g, step, H, max_iter = 1000, tol = 1.0e-1  ):
    pass

def fista_backtracking( f, x0, grad_f, prox, g, step, H, max_iter = 1000, tol = 1.0e-1  ):
    pass