import torch
class Algorithms:

    @staticmethod        
    def nesterov_exato_otimizado( f, grad_f, A, AT, x, tol = 1.0e-4, max_iter = 1000 ):
        
        y0 = x
        t0 = 1.0
        iters = [ x ]
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
            y0 = y
            x = x_trial
            iterations += 1
            iters.append( y )

        return iters        

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
    def mdbe( f, grad_f, A, x0, tol = 1.0e-4, max_iter = 1000 ):
        
        x = x0
        iterations = 0
        iters = [ x ]
        
        while f( x ) > tol and iterations < max_iter:            
            
            g_x = grad_f( x )
            r = A( g_x )

            num = torch.sum( g_x * g_x )  
            den = torch.sum( r * r )
            step = num / den

            x = x - step * g_x
            iterations += 1
            iters.append( x )

        return iters 
            

    @staticmethod
    def nesterov_fixo( f, grad_f, x, step, tol = 1.0e-4, max_iter = 1000 ):
        
        y0 = x
        t0 = 1.0
        iters = [ x ]
        iterations = 0
        
        while f( y0 ) > tol and iterations < max_iter:
            g_x = grad_f( x )
            y = x - step * g_x

            t = 0.5 * ( 1.0 + ( 1.0 + 4.0 * t0 ** 2 ) ** 0.5 )

            x_trial = y + ( ( t0 - 1.0 ) / t ) * ( y - y0 )

            t0 = t
            y0 = y
            x = x_trial
            iterations += 1
            iters.append( y )

        return iters
    
    @staticmethod
    def nesterov_backtracking( f, grad_f, x, step, sigma, beta = 0.95, tol = 1.0e-4, max_iter = 1000 ):
        y0 = x
        t0 = 1.0
        iters = [ x ]
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
            y0 = y
            x = x_trial
            iterations += 1
            iters.append( y )

        return iters

    @staticmethod
    def nesterov_exato( f, grad_f, Q, x, tol = 1.0e-4, max_iter = 1000 ):
        
        y0 = x
        t0 = 1.0
        iters = [ x ]
        iterations = 0
        
        while f( y0 ) > tol and iterations < max_iter:            
            g_x = grad_f( x )


            num = torch.sum( g_x * g_x )  
            den = torch.sum( g_x * Q( g_x ) )
            step = num / den
            
            y = x - step * g_x

            t = 0.5 * ( 1.0 + ( 1.0 + 4.0 * t0 ** 2 ) ** 0.5 )

            x_trial = y + ( ( t0 - 1.0 ) / t ) * ( y - y0 )

            t0 = t
            y0 = y
            x = x_trial
            iterations += 1
            iters.append( y )

        return iters        

    # @staticmethod
    # def conjugate( Q, b, x0, f, tol = 1.0e-4, max_iter = 1000 ):
    #     x = x0
    #     g = Q( x ) - b
    #     d = -g
    #     iters = [ x ]
    #     iterations = 0

    #     while f( x ) > tol and iterations < max_iter:
    #         Qd = Q( d )
            
    #         g_sq = torch.sum( g * g )
    #         alpha = g_sq / torch.sum( d * Qd )

    #         x = ( x + alpha * d )
            
    #         if iterations % 100 == 0:
    #             g_next = Q( x ) - b
    #             d = -g.clone()
    #         else:
    #             g_next = g + alpha * Qd 

    #         if torch.norm( g_next ) <= tol:
    #             iters.append( x )
    #             break

    #         beta = torch.sum( g_next * g_next ) / g_sq

    #         d = ( -g_next + beta * d )
    #         g = g_next

    #         iterations += 1
    #         iters.append( x )

    #     return iters
        
    @staticmethod
    def conjugate( A, AT, b, x0, f, tol = 1e-4, max_iter = 1000 ):
        x = x0
        r = b - A( x )
        s = AT( r )
        d = s.clone()
        iters = [ x.clone() ]
        iterations = 0

        while f( x ) > tol and iterations < max_iter:
            Ad = A( d )

            alpha = torch.sum( s * s ) / torch.sum( Ad * Ad )
            
            x = x + alpha * d
            
            if iterations % 100 == 0:
                r = b - A( x )
                s_next = AT( r )
            else:
                r = r - alpha * Ad
                s_next = AT( r )

            if torch.norm( s_next ) < tol:
                iters.append( x.clone() )
                break

            beta = torch.sum( s_next * s_next ) / torch.sum( s * s )
            d = s_next + beta * d
            s = s_next
            
            iterations += 1
            iters.append( x.clone() )

        return iters