import numpy as np
import pywt
class Algorithms:
    
    @staticmethod    
    def soft_thresholding( c, step, gamma ):
        return np.sign( c ) * np.maximum( np.abs( c ) - step * gamma, 0 ) # numpy faz vetorização
    
    @staticmethod
    def prox_wavelet( x, step, gamma, wavelet = 'haar', **kwargs ):
        Hx, slices = pywt.coeffs_to_array( pywt.wavedec2( x, wavelet = wavelet, mode = 'periodization' ) )
        c = Algorithms.soft_thresholding( Hx, step = step, gamma = gamma )
    
        return pywt.waverec2( pywt.array_to_coeffs( c, slices, output_format = 'wavedec2' ), wavelet = wavelet, mode = 'periodization' )

    @staticmethod
    def proj_P( p, q ):
        r = np.zeros_like( p )
        s = np.zeros_like( q )

        den = np.sqrt( p[ :, :-1 ]**2 + q[ :-1, : ]**2 )#pra nao ter que calcular duas vezes essa raíz
        
        r[ :, :-1 ] = p[ :, :-1 ] / np.maximum( 1, den )
        r[ :, -1 ] = p[ :, -1 ] / np.maximum( 1, np.abs( p[ :, -1 ] ) )

        s[ :-1, : ] = q[ :-1, : ] / np.maximum( 1, den )
        s[ -1, : ] = q[ -1, : ] / np.maximum( 1, np.abs( q[ -1, : ] ) )

        return r,s

    @staticmethod
    #gamma na verdade é gamma * step no nosso problema
    def prox_tv( b, step, gamma, max_iter, L, LT, proj_P, **kwargs ):
        p0, q0 = np.zeros( shape = ( b.shape[ 0 ] - 1, b.shape[ 1 ] ) ), np.zeros( shape = ( b.shape[ 0 ], b.shape[ 1 ] - 1 ) )
        r0, s0 = p0, q0

        t0 = 1
        iters = 0

        lbd = step * gamma

        while iters < max_iter:
            dp, dq = LT( np.maximum( 0, b - lbd * L( r0, s0 ) ) )
            p, q = proj_P( r0 + 1 / ( 8 * lbd ) * dp, s0 + 1 / ( 8 * lbd ) * dq )

            t = ( 1 + np.sqrt( 1 + 4 * t0 ** 2 ) ) / 2

            r = p + ( ( t0 - 1 ) / t ) * ( p - p0 )
            s = q + ( ( t0 - 1 ) / t ) * ( q - q0 )

            p0, q0 = p, q
            r0, s0 = r, s
            t0 = t
            iters += 1

        return np.maximum( 0, b - lbd * L( p, q ) )


    @staticmethod
    def ista_fixo( f, x0, grad_f, prox, step, gamma, max_iter = 1000, tol = 1.0e-1, **kwargs ):
        x = x0.copy()
        iterations = 0
        iters = [ x.copy() ]
        g_f = grad_f( x )        

        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x.copy()
            g_f = grad_f( x0 )        
            y = x0 - step * g_f
            x = prox( x = y, step = step, gamma = gamma, **kwargs )
            iterations += 1
            iters.append( x.copy() )
            dif = np.linalg.norm( x - x0 )
        return iters
   
    @staticmethod
    def ista_exato( f, x0, grad_f, prox, gamma, Q, max_iter = 1000, tol = 1.0e-1, **kwargs ):
        x = x0.copy()
        iterations = 0
        iters = [ x.copy() ]
        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x.copy()
            g_f = grad_f( x0 )

            num = np.sum( g_f * g_f )  
            den = np.sum( g_f * Q( g_f ) )
            step = num / den
            # if iterations % 50 == 0:
            #     print( f'O tamanho do passo para o ista com passo exato foi de: {step}' )

            y = x0 - step * g_f
            x = prox( x = y, step = step, gamma = gamma, **kwargs )
                    
            iterations += 1
            iters.append( x.copy() )
            dif = np.linalg.norm( x - x0 )

        return iters

    @staticmethod
    def fista_backtracking( f, x0, grad_f, prox, step, beta, gamma, max_iter = 1000, tol = 1.0e-1, **kwargs ):
        y0 = x0.copy()
        x = x0.copy()
        iterations = 0
        t0 = 1
        iters = [ x.copy() ]
        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x.copy()

            fy = f( y0 )
            g_y = grad_f( y0 )

            ygrad = y0 - step * g_y
            x = prox( x = ygrad, step = step, gamma = gamma, **kwargs )
   
            #while F( x ) > fy + np.sum( ( x - y0 ) * g_y ) + 1 / ( 2 * step ) * np.sum( ( x - y0 ) ** 2 ) + gamma * g( x ):
            #esse while de cima é equivalente a subtrair gamma * g(x) dos dois lados, o que resulta em:
            
            while f( x ) > fy + np.sum( ( x - y0 ) * g_y ) + 1 / ( 2 * step ) * np.sum( ( x - y0 ) ** 2 ):
                step = beta * step

                ygrad = y0 - step * g_y
                x = prox( x = ygrad, step = step, gamma = gamma, **kwargs )

            t = ( 1 + np.sqrt( 1 + 4*t0 ** 2 ) ) / 2
            y = x + ( ( t0 - 1 ) / ( t ) )  * ( x - x0 )
            iterations += 1
            iters.append( x.copy() )

            y0 = y
            t0 = t
            dif = np.linalg.norm( x - x0 )

        return iters

    @staticmethod
    def fista_fixo( f, x0, grad_f, prox, step, gamma, max_iter = 1000, tol = 1.0e-1, **kwargs ):
        y0 = x0.copy()
        x = x0.copy()
        iterations = 0
        t0 = 1
        iters = [ x.copy() ]
        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x.copy()

            ygrad = y0 - step * grad_f( y0 )
        
            x = prox( x = ygrad, step = step, gamma = gamma, **kwargs )
            t = ( 1 + np.sqrt( 1 + 4*t0 ** 2 ) ) / 2
            y = x + ( ( t0 - 1 ) / ( t ) )  * ( x - x0 )
            iterations += 1
            iters.append( x.copy() )

            y0 = y
            t0 = t
            dif = np.linalg.norm( x - x0 )

        return iters

    @staticmethod
    def fista_exato( f, x0, Q, grad_f, prox, gamma, max_iter = 1000, tol = 1.0e-1, **kwargs  ):
        y0 = x0.copy()
        x = x0.copy()
        iterations = 0
        t0 = 1
        iters = [ x.copy() ]
        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x.copy()
            g_y = grad_f( y0 )
            num = np.sum( g_y * g_y )  
            den = np.sum( g_y * Q( g_y ) )
            step = num / den
            # if iterations % 50 == 0:
            #     print( f'O tamanho do passo para o fista com passo exato foi de: {step}' )
            ygrad = y0 - step * g_y
        
            x = prox( x = ygrad, step = step, gamma = gamma, **kwargs )
            t = ( 1 + np.sqrt( 1 + 4*t0 ** 2 ) ) / 2
            y = x + ( ( t0 - 1 ) / ( t ) )  * ( x - x0 )
            
            dif = np.linalg.norm( x - x0 )


            y0 = y
            t0 = t
            iterations += 1
            iters.append( x.copy() )


        return iters

    @staticmethod
    def ista_exato_relaxado( f, x0, grad_f, prox, gamma, eta, Q, max_iter = 1000, tol = 1.0e-1, **kwargs ):
        x = x0.copy()
        iterations = 0
        iters = [ x.copy() ]
        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x.copy()
            g_f = grad_f( x0 )

            num = np.sum( g_f * g_f )  
            den = np.sum( g_f * Q( g_f ) )
            step = eta * ( num / den )

            y = x0 - step * g_f
            x = prox( x = y, step = step, gamma = gamma, **kwargs )
                    
            iterations += 1
            iters.append( x.copy() )
            dif = np.linalg.norm( x - x0 )

        return iters


    @staticmethod
    def fista_exato_relaxado( f, x0, Q, grad_f, prox, gamma, eta, max_iter = 1000, tol = 1.0e-1, **kwargs ):
        y0 = x0.copy()
        x = x0.copy()
        iterations = 0
        t0 = 1
        iters = [ x.copy() ]
        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x.copy()
            g_y = grad_f( y0 )
            num = np.sum( g_y * g_y )  
            den = np.sum( g_y * Q( g_y ) )
            step = eta * ( num / den )
            ygrad = y0 - step * g_y
        
            x = prox( x = ygrad, step = step, gamma = gamma, **kwargs )
            t = ( 1 + np.sqrt( 1 + 4*t0 ** 2 ) ) / 2
            y = x + ( ( t0 - 1 ) / ( t ) )  * ( x - x0 )
            
            dif = np.linalg.norm( x - x0 )


            y0 = y
            t0 = t
            iterations += 1
            iters.append( x.copy() )


        return iters