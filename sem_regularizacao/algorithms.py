import numpy as np

class Algorithms:

#Como nossa clase de problemas têm que f(x) é uma forma quadrática
# Busca exata se torna: lambda_k = - \frac{grad(f(xk))^Tdk}{dk^TQdk}
    
    @staticmethod
    def mdpf( f, grad_f, x0, step, tol = 1.0e-4, max_iter = 1000 ):
        x = x0
        iterations = 0
        iters = [ x ]

        while f( x ) > tol and iterations < max_iter:
            g_x = grad_f( x )

            x = x - step * g_x
            iterations += 1
            iters.append( x )

        return iters
    
        


    @staticmethod
    def mdbe( f, grad_f, Q, x0, tol = 1.0e-4, max_iter = 1000 ):
        
        x = x0
        iterations = 0
        iters = [ x ]
        
        while f( x ) > tol and iterations < max_iter:            
            
            g_x = grad_f( x )

            num = np.sum( g_x * g_x )  
            den = np.sum( g_x * Q( g_x ) )
            step = num / den

            x = x - step * g_x
            iterations += 1
            iters.append( x )

        return iters 
            
    @staticmethod
    
    ##Supondo que a nossa f é Lipschitzs e Convexa:
    def nesterov_fixo( f, grad_f, x, step, tol = 1.0e-4, max_iter = 1000 ):
        
        y0 = x.copy()
        t0 = 1.0
        iters = [ x ]
        iterations = 0
        
        while f( y0 ) > tol and iterations < max_iter:
            g_x = grad_f( x )
            y = x - step * g_x

            t = 0.5 * ( 1.0 + np.sqrt( 1.0 + 4.0 * t0 ** 2 ) )

            x_trial = y + ( ( t0 - 1.0 ) / t ) * ( y - y0 )

            t0 = t
            y0 = y
            x = x_trial
            iterations += 1
            iters.append( y )

        return iters

    def nesterov_exato( f, grad_f, Q, x, tol = 1.0e-4, max_iter = 1000 ):
        
        y0 = x.copy()
        t0 = 1.0
        iters = [ x ]
        iterations = 0
        
        while f( y0 ) > tol and iterations < max_iter:            
            g_x = grad_f( x )


            num = np.sum( g_x * g_x )  
            den = np.sum( g_x * Q( g_x ) )
            step = num / den
            
            y = x - step * g_x

            t = 0.5 * ( 1.0 + np.sqrt( 1.0 + 4.0 * t0 ** 2 ) )

            x_trial = y + ( ( t0 - 1.0 ) / t ) * ( y - y0 )

            t0 = t
            y0 = y
            x = x_trial
            iterations += 1
            iters.append( y )

        return iters        

    @staticmethod
    def nesterov_backtracking( f, grad_f, x, step, sigma, beta = 0.95, tol = 1.0e-4, max_iter = 1000 ):
        y0 = x.copy()
        t0 = 1.0
        iters = [ x ]
        iterations = 0

        while f( y0 ) > tol and iterations < max_iter:
            s = step
            g_x = grad_f( x )
            g2 = np.sum( g_x ** 2 )
            while f( x - s * g_x ) > f( x ) - sigma * s * g2:
                s = s * beta

            y = x - s * g_x

            t = 0.5 * ( 1.0 + np.sqrt( 1.0 + 4.0 * t0 ** 2 ) )

            x_trial = y + ( ( t0 - 1.0 ) / t ) * ( y - y0 )

            t0 = t
            y0 = y
            x = x_trial
            iterations += 1
            iters.append( y )

        return iters


    @staticmethod
    def conjugate( Q, b, x0, f, tol = 1.0e-4, max_iter = 1000 ):
        x = x0.copy()
        g = Q( x ) - b
        d = -g
        iters = [ x ]
        iterations = 0

        while f( x ) > tol and iterations < max_iter:
            Qd = Q( d )
            alpha = -np.sum( g * d ) / np.sum( d * Qd )

            x = x + alpha * d

            g_next = Q( x ) - b

            if np.linalg.norm( g_next ) <= tol:
                iters.append( x )
                break

            beta = np.sum( g_next * g_next ) / np.sum( g * g )

            d = -g_next + beta * d
            g = g_next

            iterations += 1
            iters.append( x )

        return iters

    @staticmethod
    def conjugate_optimized( Q, b, x0, f, tol = 1.0e-4, max_iter = 1000 ):
        x = x0.copy()
        g = Q( x ) - b
        d = -g
        iters = [ x ]
        iterations = 0

        while f( x ) > tol and iterations < max_iter:
            Qd = Q( d )
            
            g_sq = np.sum( g * g )
            alpha = g_sq / np.sum( d * Qd )

            x = x + alpha * d
            
            if iterations % 100 == 0:
                g_next = Q( x ) - b
            else:
                g_next = g + alpha * Qd 

            if np.linalg.norm( g_next ) <= tol:
                iters.append( x )
                break

            beta = np.sum( g_next * g_next ) / g_sq

            d = -g_next + beta * d
            g = g_next

            iterations += 1
            iters.append( x )

        return iters