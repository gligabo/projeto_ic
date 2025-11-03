import numpy as np
import pywt
class Algorithms:
    
    @staticmethod    
    def soft_thresholding( c, step, gamma ):
        return np.sign( c ) * np.maximum( np.abs( c ) - step * gamma, 0 ) # numpy faz vetorização
    

    @staticmethod
    def prox_wavelet( x, step, gamma, wavelet = 'haar' ):
        Hx, slices = pywt.coeffs_to_array( pywt.wavedec2( x, wavelet = wavelet, mode = 'periodization' ) )
        c = Algorithms.soft_thresholding( Hx, step = step, gamma = gamma )
    
        return pywt.waverec2( pywt.array_to_coeffs( c, slices, output_format = 'wavedec2' ), wavelet = wavelet, mode = 'periodization' )

    def prox_tv(  ):
        pass

    @staticmethod
    #note que por enquanto só funciona p/ transformada wavelet
    def ista_fixo( f, x0, grad_f, prox, step, gamma, max_iter = 1000, tol = 1.0e-1, wavelet = 'haar' ):
        x = x0.copy()
        iterations = 0
        iters = [ x.copy() ]
        g_f = grad_f( x )        

        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x.copy()
            g_f = grad_f( x0 )        
            y = x0 - step * g_f
            x = prox( x = y, step = step, gamma = gamma, wavelet = wavelet )
            iterations += 1
            iters.append( x.copy() )
            dif = np.linalg.norm( x - x0 )
        return iters
   
    @staticmethod
    def ista_exato( f, x0, grad_f, prox, gamma, Q, max_iter = 1000, tol = 1.0e-1, wavelet = 'haar' ):
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
            if iterations % 50 == 0:
                print( f'O tamanho do passo para o ista com passo exato foi de: {step}' )

            y = x0 - step * g_f
            x = prox( x = y, step = step, gamma = gamma, wavelet = wavelet )
                    
            iterations += 1
            iters.append( x.copy() )
            dif = np.linalg.norm( x - x0 )

        return iters


    # def ista_backtracking( f, x0, grad_f, prox, step, H, max_iter = 1000, tol = 1.0e-1  ):
    #     pass

    @staticmethod
    def fista_fixo( f, x0, grad_f, prox, step, gamma, max_iter = 1000, tol = 1.0e-1, wavelet = 'haar' ):
        y0 = x0.copy()
        x = x0.copy()
        iterations = 0
        t0 = 1
        iters = [ x.copy() ]
        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x.copy()

            ygrad = y0 - step * grad_f( y0 )
        
            x = prox( x = ygrad, step = step, gamma = gamma, wavelet = wavelet )
            t = ( 1 + np.sqrt( 1 + 4*t0 ** 2 ) ) / 2
            y = x + ( ( t0 - 1 ) / ( t ) )  * ( x - x0 )
            iterations += 1
            iters.append( x.copy() )

            y0 = y
            t0 = t
            dif = np.linalg.norm( x - x0 )

        return iters

    @staticmethod
    def fista_exato( f, x0, Q, grad_f, prox, gamma, max_iter = 1000, tol = 1.0e-1, wavelet = 'haar'  ):
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
            if iterations % 50 == 0:
                print( f'O tamanho do passo para o fista com passo exato foi de: {step}' )
            ygrad = y0 - step * g_y
        
            x = prox( x = ygrad, step = step, gamma = gamma, wavelet = wavelet )
            t = ( 1 + np.sqrt( 1 + 4*t0 ** 2 ) ) / 2
            y = x + ( ( t0 - 1 ) / ( t ) )  * ( x - x0 )
            
            dif = np.linalg.norm( x - x0 )


            y0 = y
            t0 = t
            iterations += 1
            iters.append( x.copy() )


        return iters


    # def fista_backtracking( f, x0, grad_f, prox, step, H, max_iter = 1000, tol = 1.0e-1  ):
    #     pass

