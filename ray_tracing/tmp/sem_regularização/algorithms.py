import torch
class Algorithms:

    @staticmethod        
    def nesterov_exato_otimizado( f, grad_f, A, AT, x, tol = 1.0e-4, max_iter = 1000 ):
        
        y0 = x.clone().detach()
        t0 = 1.0
        iters = [ x.clone().detach() ]
        iterations = 0
        
        while f( y0 ) > tol and iterations < max_iter:            
            g_x = grad_f( x )
            
            r = A( g_x )

            num = torch.sum( g_x * g_x )
            den = torch.sum( r * r )
            step = num / den
            
            y = x - step * g_x

            t = 0.5 * ( 1.0 + ( 1.0 + 4.0 * t0 ** 2 ) ** 0.5 )

            x_trial = y + ( ( t0 - 1.0 ) / t ) * ( y - y0 )

            t0 = t
            y0 = y.clone().detach()
            x = x_trial.clone().detach()
            iterations += 1
            iters.append( y.clone().detach() )

        return iters        

    @staticmethod
    def mdpf( f, grad_f, x0, step, tol = 1.0e-4, max_iter = 1000 ):
        x = x0.clone().detach()
        iterations = 0
        iters = [ x.clone().detach() ]

        while f( x ) > tol and iterations < max_iter:
            g_x = grad_f( x )

            x = x - step * g_x
            iterations += 1
            iters.append( x.clone().detach() )

        return iters


    @staticmethod
    def mdbe( f, grad_f, Q, x0, tol = 1.0e-4, max_iter = 1000 ):
        
        x = x0.clone().detach()
        iterations = 0
        iters = [ x.clone().detach() ]
        
        while f( x ) > tol and iterations < max_iter:            
            
            g_x = grad_f( x )

            num = torch.sum( g_x * g_x )  
            den = torch.sum( g_x * Q( g_x ) )
            step = num / den

            x = x - step * g_x
            iterations += 1
            iters.append( x.clone().detach() )

        return iters 
            

    @staticmethod
    def nesterov_fixo( f, grad_f, x, step, tol = 1.0e-4, max_iter = 1000 ):
        
        y0 = x.clone().detach()
        t0 = 1.0
        iters = [ x.clone().detach() ]
        iterations = 0
        
        while f( y0 ) > tol and iterations < max_iter:
            g_x = grad_f( x )
            y = x - step * g_x

            t = 0.5 * ( 1.0 + ( 1.0 + 4.0 * t0 ** 2 ) ** 0.5 )

            x_trial = y + ( ( t0 - 1.0 ) / t ) * ( y - y0 )

            t0 = t
            y0 = y.clone().detach()
            x = x_trial.clone().detach()
            iterations += 1
            iters.append( y.clone().detach() )

        return iters
    
    @staticmethod
    def nesterov_backtracking( f, grad_f, x, step, sigma, beta = 0.95, tol = 1.0e-4, max_iter = 1000 ):
        y0 = x.clone().detach()
        t0 = 1.0
        iters = [ x.clone().detach() ]
        iterations = 0

        while f( y0 ) > tol and iterations < max_iter:
            s = step
            g_x = grad_f( x )
            g2 = torch.sum( g_x ** 2 )
            while f( x - s * g_x ) > f( x ) - sigma * s * g2:
                s = s * beta

            y = x - s * g_x

            t = 0.5 * ( 1.0 + ( 1.0 + 4.0 * t0 ** 2 ) ** 0.5 )

            x_trial = y + ( ( t0 - 1.0 ) / t ) * ( y - y0 )

            t0 = t
            y0 = y.clone().detach()
            x = x_trial.clone().detach()
            iterations += 1
            iters.append( y.clone().detach() )

        return iters

    @staticmethod
    def conjugate( Q, b, x0, f, tol = 1.0e-4, max_iter = 1000 ):
        x = x0.clone().detach()
        g = Q( x ) - b
        d = -g.clone().detach()
        iters = [ x.clone().detach() ]
        iterations = 0

        while f( x ) > tol and iterations < max_iter:
            Qd = Q( d )
            
            g_sq = torch.sum( g * g )
            alpha = g_sq / torch.sum( d * Qd )

            x = ( x + alpha * d ).clone().detach()
            
            if iterations % 100 == 0:
                g_next = Q( x ) - b
            else:
                g_next = g + alpha * Qd 

            if torch.norm( g_next ) <= tol:
                iters.append( x )
                break

            beta = torch.sum( g_next * g_next ) / g_sq

            d = ( -g_next + beta * d ).clone().detach()
            g = g_next.clone().detach()

            iterations += 1
            iters.append( x.clone().detach() )

        return iters