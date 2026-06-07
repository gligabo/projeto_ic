import torch
import pywt

class Algorithms:
    
    @staticmethod    
    def soft_thresholding( c, step, gamma ):
        return torch.sign( c ) * torch.clamp( torch.abs( c ) - step * gamma, min = 0.0 ) # torch faz vetorização
    
    @staticmethod
    def prox_wavelet( x, step, gamma, wavelet = 'haar', **kwargs ):
        x_np = x.detach().cpu().numpy()
        coeffs = pywt.wavedec2( x_np, wavelet = wavelet, mode = 'periodization' )
        
        arr_np, slices = pywt.coeffs_to_array( coeffs )

        Hx = torch.from_numpy( arr_np )

        c = Algorithms.soft_thresholding( Hx, step = step, gamma = gamma )
    
        return pywt.waverec2( pywt.array_to_coeffs( c, slices, output_format = 'wavedec2' ), wavelet = wavelet, mode = 'periodization' )

    def prox_wavelet(x, step, gamma, wavelet='haar', **kwargs):
        orig_device = x.device
        orig_dtype = x.dtype
        
        x_np = x.detach().cpu().numpy()
        
        coeffs = pywt.wavedec2( x_np, wavelet = wavelet, mode = 'periodization' )
        arr_np, slices = pywt.coeffs_to_array( coeffs )

        Hx = torch.from_numpy( arr_np )

        c = Algorithms.soft_thresholding( Hx, step = step, gamma = gamma )
    
        c_np = c.numpy()
        
        coeffs_rec = pywt.array_to_coeffs( c_np, slices, output_format = 'wavedec2' )
        x_rec_np = pywt.waverec2( coeffs_rec, wavelet = wavelet, mode = 'periodization' )
        
        return torch.tensor( x_rec_np, device = orig_device, dtype = orig_dtype )


    @staticmethod
    def proj_P( p, q ):
        r = torch.zeros_like( p )
        s = torch.zeros_like( q )

        den = ( p[ :, :-1 ]**2 + q[ :-1, : ]**2 ) ** 0.5#pra nao ter que calcular duas vezes essa raíz
        
        r[ :, :-1 ] = p[ :, :-1 ] / torch.maximum( 1, den )
        r[ :, -1 ] = p[ :, -1 ] / torch.maximum( 1, torch.abs( p[ :, -1 ] ) )

        s[ :-1, : ] = q[ :-1, : ] / torch.maximum( 1, den )
        s[ -1, : ] = q[ -1, : ] / torch.maximum( 1, torch.abs( q[ -1, : ] ) )

        return r,s

    @staticmethod
    def prox_tv( b, step, gamma, max_iter_tv, L, LT, proj_P, **kwargs ):
        p0, q0 = torch.zeros( shape = ( b.shape[ 0 ] - 1, b.shape[ 1 ] ) ), torch.zeros( shape = ( b.shape[ 0 ], b.shape[ 1 ] - 1 ) )
        r0, s0 = p0, q0

        t0 = 1
        iters = 0

        lbd = step * gamma

        while iters < max_iter_tv:
            dp, dq = LT( torch.maximum( 0, b - lbd * L( r0, s0 ) ) )
            p, q = proj_P( r0 + 1 / ( 8 * lbd ) * dp, s0 + 1 / ( 8 * lbd ) * dq )

            t = ( 1 + ( 1 + 4 * t0 ** 2 ) ** 0.5 ) / 2

            r = p + ( ( t0 - 1 ) / t ) * ( p - p0 )
            s = q + ( ( t0 - 1 ) / t ) * ( q - q0 )

            p0, q0 = p, q
            r0, s0 = r, s
            t0 = t
            iters += 1

        return torch.maximum( 0, b - lbd * L( p, q ) )


    @staticmethod
    def ista_fixo( f, x0, grad_f, prox, step, gamma, max_iter = 1000, tol = 1.0e-1, **kwargs ):
        x = x0
        iterations = 0
        iters = [ x ]
        g_f = grad_f( x )        

        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x
            g_f = grad_f( x0 )        
            y = x0 - step * g_f
            x = prox( y, step = step, gamma = gamma, **kwargs )
            iterations += 1
            iters.append( x )
            dif = torch.linalg.norm( x - x0 )
        return iters
   
    @staticmethod
    def ista_exato( f, x0, grad_f, prox, gamma, A, max_iter = 1000, tol = 1.0e-1, **kwargs ):
        x = x0
        iterations = 0
        iters = [ x ]
        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x
            g_f = grad_f( x0 )
            r = A( g_f )
            num = torch.sum( g_f * g_f )  
            den = torch.sum( r * r )
            step = ( num / den ).item()
            # if iterations % 50 == 0:
            #     print( f'O tamanho do passo para o ista com passo exato foi de: {step}' )

            y = x0 - step * g_f
            x = prox( y, step = step, gamma = gamma, **kwargs )
                    
            iterations += 1
            iters.append( x )
            dif = torch.linalg.norm( x - x0 )

        return iters

    @staticmethod
    def fista_backtracking( f, x0, grad_f, prox, step, beta, gamma, max_iter = 1000, tol = 1.0e-1, **kwargs ):
        y0 = x0
        x = x0
        iterations = 0
        t0 = 1
        iters = [ x ]
        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x

            fy = f( y0 )
            g_y = grad_f( y0 )

            ygrad = y0 - step * g_y
            x = prox( ygrad, step = step, gamma = gamma, **kwargs )
   
            #while F( x ) > fy + np.sum( ( x - y0 ) * g_y ) + 1 / ( 2 * step ) * np.sum( ( x - y0 ) ** 2 ) + gamma * g( x ):
            #esse while de cima é equivalente a subtrair gamma * g(x) dos dois lados, o que resulta em:
            
            while f( x ) > fy + torch.sum( ( x - y0 ) * g_y ) + 1 / ( 2 * step ) * torch.sum( ( x - y0 ) ** 2 ):
                step = beta * step

                ygrad = y0 - step * g_y
                x = prox( ygrad, step = step, gamma = gamma, **kwargs )

            t = ( 1 + ( 1 + 4*t0 ** 2 ) ** 0.5 ) / 2
            y = x + ( ( t0 - 1 ) / ( t ) )  * ( x - x0 )
            iterations += 1
            iters.append( x )

            y0 = y
            t0 = t
            dif = torch.linalg.norm( x - x0 )

        return iters

    @staticmethod
    def fista_fixo( f, x0, grad_f, prox, step, gamma, max_iter = 1000, tol = 1.0e-1, **kwargs ):
        y0 = x0
        x = x0
        iterations = 0
        t0 = 1
        iters = [ x ]
        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x

            ygrad = y0 - step * grad_f( y0 )
        
            x = prox( ygrad, step = step, gamma = gamma, **kwargs )
            t = ( 1 + ( 1 + 4*t0 ** 2 ) ** 0.5 ) / 2
            y = x + ( ( t0 - 1 ) / ( t ) )  * ( x - x0 )
            iterations += 1
            iters.append( x )

            y0 = y
            t0 = t
            dif = torch.linalg.norm( x - x0 )

        return iters

    @staticmethod
    def fista_exato( f, x0, A, grad_f, prox, gamma, max_iter = 1000, tol = 1.0e-1, **kwargs  ):
        y0 = x0
        x = x0
        iterations = 0
        t0 = 1
        iters = [ x ]
        dif = float( 'inf' )

        while dif > tol and iterations < max_iter:
            x0 = x
            g_y = grad_f( y0 )
            r = A( g_y )
            num = torch.sum( g_y * g_y )  
            den = torch.sum( r * r )
            step = ( num / den ).item()
            # if iterations % 50 == 0:
            #     print( f'O tamanho do passo para o fista com passo exato foi de: {step}' )
            ygrad = y0 - step * g_y
        
            x = prox( ygrad, step = step, gamma = gamma, **kwargs )
            t = ( 1 + ( 1 + 4*t0 ** 2 ) ** 0.5 ) / 2
            y = x + ( ( t0 - 1 ) / ( t ) )  * ( x - x0 )
            
            dif = torch.linalg.norm( x - x0 )


            y0 = y
            t0 = t
            iterations += 1
            iters.append( x )
        return iters