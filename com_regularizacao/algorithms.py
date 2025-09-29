import numpy as np
class Algorithms():
    
    @staticmethod    
    def soft_thresholding( c, step, gamma ):
        return np.sign( c ) * np.maximum( np.abs( c ) - step * gamma, 0 ) # numpy faz vetorização
    
    @staticmethod
    def prox_wavelet( H, HT, x, step, gamma ):
        c = H( x )
        S = Algorithms.soft_thresholding( c, step, gamma )
        return HT( S )

    def prox_tv(  ):
        pass

    @staticmethod
    #note que por enquanto só funciona p/ transformada wavelet
    def ista_fixo( f, x0, grad_f, prox, step, gamma, H, HT, max_iter = 1000, tol = 1.0e-1  ):
        x = x0.copy()
        iterations = 0
        iters = [ x.copy() ]
        while f( x ) > tol and iterations < max_iter:
            g_f = grad_f( x )        
            y = x - step * g_f
            x = prox( H = H, HT = HT, x = y, step = step, gamma = gamma )
            iterations += 1
            iters.append( x.copy() )

        return iters
   
    @staticmethod
    def ista_exato( f, x0, grad_f, prox, gamma, Q, H, HT, max_iter = 1000, tol = 1.0e-1  ):
        x = x0.copy()
        iterations = 0
        iters = [ x.copy() ]
        while f( x ) > tol and iterations < max_iter:
            g_f = grad_f( x )

            num = np.sum( g_f * g_f )  
            den = np.sum( g_f * Q( g_f ) )
            step = num / den
            
            y = x - step * g_f
            x = prox( H = H, HT = HT, x = y, step = step, gamma = gamma )
                    
            iterations += 1
            iters.append( x.copy() )

        return iters


    # def ista_backtracking( f, x0, grad_f, prox, step, H, max_iter = 1000, tol = 1.0e-1  ):
    #     pass

    @staticmethod
    def fista_fixo( f, x0, grad_f, prox, step, gamma, H, HT, max_iter = 1000, tol = 1.0e-1  ):
        y0 = x0.copy()
        x = x0.copy()
        iterations = 0
        t0 = 1
        iters = [ x.copy() ]

        while f( x ) > tol and iterations < max_iter:
        
            ygrad = y0 - step * grad_f( y0 )
        
            x = prox( H = H, HT = HT, x = ygrad, step = step, gamma = gamma )
            t = ( 1 + np.sqrt( 1 + 4*t0 ** 2 ) ) / 2
            y = x + ( ( t0 - 1 ) / ( t ) )  * ( x - x0 )
            iterations += 1
            iters.append( x.copy() )

            x0 = x
            y0 = y
            t0 = t

        return iters

    @staticmethod
    def fista_exato( f, x0, Q, grad_f, prox, H, HT, gamma, max_iter = 1000, tol = 1.0e-1  ):
        y0 = x0.copy()
        x = x0.copy()
        iterations = 0
        t0 = 1
        iters = [ x.copy() ]

        while f( x ) > tol and iterations < max_iter:
            g_y = grad_f( y0 )
            num = np.sum( g_y * g_y )  
            den = np.sum( g_y * Q( g_y ) )
            step = num / den

            ygrad = y0 - step * g_y
        
            x = prox( H = H, HT = HT, x = ygrad, step = step, gamma = gamma )
            t = ( 1 + np.sqrt( 1 + 4*t0 ** 2 ) ) / 2
            y = x + ( ( t0 - 1 ) / ( t ) )  * ( x - x0 )
            iterations += 1
            iters.append( x.copy() )

            x0 = x
            y0 = y
            t0 = t
        return iters


    # def fista_backtracking( f, x0, grad_f, prox, step, H, max_iter = 1000, tol = 1.0e-1  ):
    #     pass

