from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def analytical_u(x,y,t,**params):
    """
    Точное решение u. Было дано в условии лабораторной
    x, y,t : numpy array (or float)
    output: numpy array (or float)
    может принимать дополнительные параметры:
    d = 1 (по умолчанию)
    """

    d = params.get('d') or 1

    return - np.exp(-t)* np.exp(2*y/d) *np.sin(2*x/d)


def analytical_v(x,y,t,**params):
    """
    Точное решение v. Было дано в условии лабораторной
    x, y,t : numpy array (or float)
    output: numpy array (or float)
    может принимать дополнительные параметры:
    d = 1 (по умолчанию)
    """

    d = params.get('d') or 1

    return np.exp(-t)* np.exp(2*y/d) *np.cos(2*x/d)

def analytical_p(x,y,t,**params):
    """
    Точное решение p. Было дано в условии лабораторной
    x, y,t : numpy array (or float)
    output: numpy array (or float)
    может принимать дополнительные параметры:
    d = 1 (по умолчанию)
    rho = 1 (по умолчанию)
    """

    d = params.get('d') or 1
    rho = params.get('rho') or 1

    return d/2 * rho * np.exp(-t) * np.exp(2*y/d) *np.cos (2*x/d)

def absolute_error(analytical_decision, method_decision):
    """
    Принимает 2 сетки решений, возвращает число
    """
    return np.max(np.abs(analytical_decision - method_decision))


class BoundaryConditions_first_type:
    """
    Кравевые условия первого рода 
    """
    def __init__(self,x_lim,y_lim, t_0, analytical_u, analytical_v, analytical_p, **params):
        self.x_lim = x_lim # [x_min, x_max]
        self.y_lim = y_lim
        self.t_0 = t_0
        # функции аналитического решения
        self._analytical_u = analytical_u
        self._analytical_v = analytical_v
        self._analytical_p = analytical_p
        # параметры для аналитического решения
        self._params = params

    def u_x_left_cond(self,y,t):
        _x = np.ones(y.shape)* self.x_lim[0]
        return self._analytical_u(_x, y, t, **self._params)

    def u_x_right_cond(self,y,t):
        _x = np.ones(y.shape)* self.x_lim[1]
        return self._analytical_u(_x, y, t, **self._params)

    def u_y_left_cond(self,x,t):
        _y = np.ones(x.shape)* self.y_lim[0]
        return self._analytical_u(x, _y, t, **self._params)

    def u_y_right_cond(self,x,t):
        _y = np.ones(x.shape)* self.y_lim[1]
        return self._analytical_u(x, _y, t, **self._params)
    
    def u_time_cond(self,x,y):
        xv, yv = np.meshgrid(x,y)
        return self._analytical_u(xv, yv, self.t_0, **self._params)


    def v_x_left_cond(self,y,t):
        _x = np.ones(y.shape)* self.x_lim[0]
        return self._analytical_v(_x, y, t, **self._params)

    def v_x_right_cond(self,y,t):
        _x = np.ones(y.shape)* self.x_lim[1]
        return self._analytical_v(_x, y, t, **self._params)

    def v_y_left_cond(self,x,t):
        _y = np.ones(x.shape)* self.y_lim[0]
        return self._analytical_v(x, _y, t, **self._params)

    def v_y_right_cond(self,x,t):
        _y = np.ones(x.shape)* self.y_lim[1]
        return self._analytical_v(x, _y, t, **self._params)
    
    def v_time_cond(self,x,y):
        xv, yv = np.meshgrid(x,y)
        return self._analytical_v(xv, yv, self.t_0, **self._params)


    def p_x_left_cond(self,y,t):
        _x = np.ones(y.shape)* self.x_lim[0]
        return self._analytical_p(_x, y, t, **self._params)

    def p_x_right_cond(self,y,t):
        _x = np.ones(y.shape)* self.x_lim[1]
        return self._analytical_p(_x, y, t, **self._params)

    def p_y_left_cond(self,x,t):
        _y = np.ones(x.shape)* self.y_lim[0]
        return self._analytical_p(x, _y, t, **self._params)

    def p_y_right_cond(self,x,t):
        _y = np.ones(x.shape)* self.y_lim[1]
        return self._analytical_p(x, _y, t, **self._params)
    
    def p_time_cond(self,x,y):
        xv, yv = np.meshgrid(x,y)
        return self._analytical_p(xv, yv, self.t_0, **self._params)

    def cond(self, x,y, t,direction, type_is, cond_axis ):
        """
        all right/left , u/v/p cond in one
        direction = 'r' or 'l' 
        (right/left)
        type_is = 'u' or 'v' or 'p'
        cond_axis = 'x' or 'y'
        """
        if type_is == 'u':
            if direction == 'r':
                if cond_axis == 'x':
                    return self.u_x_right_cond(y,t)
                elif cond_axis == 'y':
                    return self.u_y_right_cond(x,t)
            elif direction == 'l':
                if cond_axis == 'x':
                    return self.u_x_left_cond(y,t)
                elif cond_axis == 'y':
                    return self.u_y_left_cond(x,t)
            else:
                raise ValueError("direction have to be equal 'r' or 'l'")
        elif type_is == 'v':
            if direction == 'r':
                if cond_axis == 'x':
                    return self.v_x_right_cond(y,t)
                elif cond_axis == 'y':
                    return self.v_y_right_cond(x,t)
            elif direction == 'l':
                if cond_axis == 'x':
                    return self.v_x_left_cond(y,t)
                elif cond_axis == 'y':
                    return self.v_y_left_cond(x,t)
            else:
                raise ValueError("direction have to be equal 'r' or 'l'")
        elif type_is == 'p':
            if direction == 'r':
                if cond_axis == 'x':
                    return self.p_x_right_cond(y,t)
                elif cond_axis == 'y':
                    return self.p_y_right_cond(x,t)
            elif direction == 'l':
                if cond_axis == 'x':
                    return self.p_x_left_cond(y,t)
                elif cond_axis == 'y':
                    return self.p_y_left_cond(x,t)
            else:
                raise ValueError("direction have to be equal 'r' or 'l'")
        else:
            raise ValueError("type_is have to be equal 'u' or 'v' or 'p'")
        raise ValueError("cond_axis have to be equal 'x' or 'y'")

    def u_my_cond(self, xv,yv,t):
        return self._analytical_u(xv,yv,t) 
    def v_my_cond(self, xv,yv,t):
        return self._analytical_v(xv,yv,t) 
    def p_my_cond(self, xv,yv,t):
        return self._analytical_p(xv,yv,t) 

    


class Method:
    def __init__(self, x, y, bound_cond, hx, hy, tau, w = 1.1, relax_steps = 10, **params):
        """
        bound_cond - это представитель класса BoundaryConditions_first_type
        Важно заметить, что x,y подается как массив значений без краевых точек!!
        в Данном случае все краевые точки считаются из bound_cond
        w = 1.5 параметр релаксации 1 <= w <=2
        в params записаны переменные 
        rho, d, visc
        """
        self.x = x
        self.y = y

        self._params = params
        self.rho = self._params.get('rho') or 1
        self.visc = self._params.get('visc') or 1
        
        self.w = w
        self.relax_steps = relax_steps

        self.bound_cond = bound_cond

        # Начальное значение времени
        self.t = self.bound_cond.t_0
        # Количество обновлений (шагов по времени)
        self.count = 0

        # Шаг по пространству для x и y
        self.hx = hx
        self.hy = hy
        # Шаг по времени
        self.tau = tau

        self._xv, self._yv = np.meshgrid(self.x,self.y)
        # состояние сеток в данный момент
        # на начальном шаге инициализируем из краевых условий
        self.u = self.bound_cond.u_time_cond(self.x,self.y)
        self.v = self.bound_cond.v_time_cond(self.x, self.y)
        self.p = self.bound_cond.p_time_cond(self.x,self.y)
    
    def Get_u(self):
        """
        Возвращает сетку составляющей скорости по x в текущий момент
        """
        return self.u

    def Get_v(self):
        """
        Возвращает сетку составляющей скорости по y в текущий момент
        """
        return self.v

    def Get_p(self):
        """
        Возвращает сетку давления в текущий момент
        """
        return self.p

    def __grid_left_right(self, grid, type_is , direction, cond_axis, t = None):
        """
        type is = 'p' or 'v' or 'u'
        direction = 'r' or 'l' 
        (right/left)
        cond_axis = 'x' or 'y'
        """

        # if type_is =='v':
        #     grid = self.v.copy()
        # elif type_is == 'u':
        #     grid= self.u.copy()
        # else:
        #     grid = self.p.copy()

        if t is None:
            t = self.t

        if cond_axis == 'x':
            if direction == 'l':
                grid = np.hstack([self.bound_cond.cond(self.x,self.y, t,direction, type_is, cond_axis).reshape(-1,1),
                    grid])[:,:-1]
            else:
                grid = np.hstack([grid, self.bound_cond.cond(self.x,self.y, t,direction, type_is, cond_axis).reshape(-1,1)])[:,1:]
        
        else:
            if direction == 'l':
                grid = np.vstack([self.bound_cond.cond(self.x,self.y, t,direction, type_is, cond_axis).reshape(1,-1), grid])[:-1,:]
            else:
                grid = np.vstack([grid, self.bound_cond.cond(self.x,self.y, t,direction, type_is, cond_axis).reshape(1,-1)])[1:,:]
        # print(grid.shape)
        return grid

    def __D_left_right(self, direction, cond_axis):
        """
        direction = 'r' or 'l' 
        (right/left)
        cond_axis = 'x' or 'y'

        return new D which is shifted to right or left along the cond_axis (x or y)
        """
        pass

    def __RE(self):
        """
        Reynolds number
        """
        pass

    def __D(self):
        """
        D из описания лабораторной работы
        """
        D = (self.__grid_left_right(self.u,'u',direction= 'r', cond_axis= 'x') - 
                self.__grid_left_right(self.u,'u',direction= 'l', cond_axis= 'x'))/(2*self.hx)  + \
            (self.__grid_left_right(self.v,'v',direction='r',cond_axis= 'y') - 
                self.__grid_left_right(self.v,'v',direction='l',cond_axis='y'))/(2*self.hy)

        return D

    def __S(self):
        """
        S из описания лабораторной работы (правая часть при вычислении давления)
        """
        S = self.__D()/self.tau
        
        # # тут вопросы
        # D = self.__D()
        # S = S + 1/self.__RE() * (self.__D_left_right( direction='r',cond_axis='x') - 2*D +  
        #                         self.__D_left_right(direction='l', cond_axis='x'))/self.hx**2 + \
        #                     (self.__D_left_right(direction='r',cond_axis='y') - 2*D +
        #                     self.__D_left_right(direction='l', cond_axis= 'y'))/self.hy**2

        return S


    def _update_u(self):
        
        p_term  = 1/self.rho * (self.__grid_left_right(self.p, type_is= 'p', direction= 'r', cond_axis='x') - 
                self.__grid_left_right(self.p, type_is='p', direction='l', cond_axis='x'))/self.hx/2

        new_u_xr  = self.__grid_left_right(self.u, type_is='u', direction='r', cond_axis='x')
        new_u_xl = self.__grid_left_right(self.u, type_is='u', direction = 'l', cond_axis = 'x')
        new_u_yr  = self.__grid_left_right(self.u, type_is='u', direction='r', cond_axis='y')
        new_u_yl = self.__grid_left_right(self.u, type_is='u', direction = 'l', cond_axis = 'y')
        
        explicit_u = self.u - self.tau*p_term + self.tau*self.visc* ( (new_u_xr -2*self.u+ new_u_xl)/self.hx**2 + (new_u_yr - 2*self.u + new_u_yl)/self.hy**2)


        # explicit_u = self.u  - self.tau*p_term + self.tau*self.visc *( (self.__grid_left_right(self.u,type_is='u',direction='r',cond_axis='x')- 2* self.u + 
        #                 self.__grid_left_right(self.u,type_is='u',direction='l',cond_axis='x'))/self.hx**2 +
        #         (self.__grid_left_right(self.u,type_is='u', direction='r',cond_axis='y') - 2*self.u +  
        #             self.__grid_left_right(self.u, type_is='u',direction='l',cond_axis='y'))/self.hy**2)

        new_u_xr = self.__grid_left_right(explicit_u, type_is='u', direction='r', cond_axis='x', t = self.t+self.tau)
        new_u_xl = self.__grid_left_right(explicit_u, type_is='u', direction='l', cond_axis='x', t = self.t+self.tau)
        new_u_yr = self.__grid_left_right(explicit_u, type_is='u', direction='r', cond_axis='y', t = self.t+self.tau)
        new_u_yl = self.__grid_left_right(explicit_u, type_is='u', direction='l', cond_axis='y', t = self.t+self.tau)

        implicit_u = ( self.u - self.tau*p_term + self.tau*self.visc* ( (new_u_xr+new_u_xl)/self.hx**2 + (new_u_yl+new_u_yr)/self.hy**2))/\
            (1+ self.tau*2*self.visc*(1/self.hx**2 + 1/self.hy**2))

        for i in range(self.u.shape[0]):
            for j in range(self.u.shape[1]):
                if (i+j+self.count) % 2 ==0:
                    self.u[i,j] = explicit_u[i,j]
                else:
                    self.u[i,j] = implicit_u[i,j]
        

    def _update_v(self):
    
        p_term  = 1/self.rho * (self.__grid_left_right(self.p, type_is= 'p', direction= 'r', cond_axis='y') - 
                self.__grid_left_right(self.p, type_is='p', direction='l', cond_axis='y'))/self.hy/2

        new_v_xr = self.__grid_left_right(self.v,type_is='v',direction='r',cond_axis='x')
        new_v_xl = self.__grid_left_right(self.v,type_is='v', direction= 'l', cond_axis='x')
        new_v_yr = self.__grid_left_right(self.v,type_is='v', direction= 'r', cond_axis='y')
        new_v_yl = self.__grid_left_right(self.v,type_is='v', direction= 'l', cond_axis='y')

        explicit_v = self.v - self.tau*p_term + self.tau*self.visc * ( (new_v_xr - 2*self.v + new_v_xl)/self.hx**2 + (new_v_yr - 2*self.v + new_v_yl)/self.hy**2)

        new_v_xr = self.__grid_left_right(explicit_v,type_is='v',direction='r',cond_axis='x', t = self.t+self.tau)
        new_v_xl = self.__grid_left_right(explicit_v,type_is='v', direction= 'l', cond_axis='x', t = self.t+self.tau)
        new_v_yr = self.__grid_left_right(explicit_v,type_is='v', direction= 'r', cond_axis='y', t = self.t+self.tau)
        new_v_yl = self.__grid_left_right(explicit_v,type_is='v', direction= 'l', cond_axis='y', t = self.t+self.tau)


        implicit_v = (self.v - self.tau*p_term + self.tau*self.visc *(  (new_v_xl + new_v_xr)/self.hx**2  + (new_v_yl + new_v_yr)/self.hy**2)) /\
                    (1+ self.tau*2*self.visc*(1/self.hx**2 + 1/self.hy**2))

        
        for i in range(self.v.shape[0]):
            for j in range(self.v.shape[1]):
                if (i+j+self.count) % 2 ==0:
                    self.v[i,j] = explicit_v[i,j]
                else:
                    self.v[i,j] = implicit_v[i,j]

    def _update_p(self, steps= 10):
        """
        Метод верхней релаксации
        """
        p_now = self.p.copy()
    
        
        beta = self.hx/self.hy

        _xi = ( ( np.cos(np.pi / (self.bound_cond.x_lim[1] -self.bound_cond.x_lim[0])/self.hx) + beta**2 *np.cos(np.pi / (self.bound_cond.y_lim[1] -self.bound_cond.y_lim[0])/self.hy)  )/(1 + beta**2))**2

        self.w = 2* ( 1 - np.sqrt(1 - _xi))/_xi


        div = 2*(1+ beta**2)
        S = self.__S()

        print('S max:', np.max(np.abs(S)))
        left_zero_term  = self.w/div*self.bound_cond.p_x_left_cond(self.y,self.t)

        for _ in range(steps):
            res = []
            p_now_xr = self.__grid_left_right(p_now, type_is= 'p', direction='r',cond_axis= 'x')
            p_now_yr = self.__grid_left_right(p_now, type_is= 'p', direction='r',cond_axis= 'y')

            right_part = p_now + self.w / div *(p_now_xr + beta**2 * p_now_yr - self.hx**2 * S - div*p_now)

            p_before = self.bound_cond.p_y_left_cond(self.x,self.t)
            
            for j in range(p_now.shape[0]):
                A = np.eye(p_now.shape[1]) + np.diagflat(np.ones(p_now.shape[1]-1)*-1*self.w / div , k = -1)
                b = right_part[j] + beta**2 * self.w / div * p_before
                b[0] += left_zero_term[j]

                p_before = np.linalg.solve(A,b)
                res.append(p_before)
            
            p_now = np.array(res)

        self.p =p_now
        

    def __update_p(self):
        print('--------------')
        new_p = self.p.copy()

        beta = self.hx/self.hy

        _xi = ( ( np.cos(np.pi / (self.bound_cond.x_lim[1] -self.bound_cond.x_lim[0])/self.hx) + beta**2 *np.cos(np.pi / (self.bound_cond.y_lim[1] -self.bound_cond.y_lim[0])/self.hy)  )/(1 + beta**2))**2

        self.w = 2* ( 1 - np.sqrt(1 - _xi))/_xi
        
        print(self.w)

        div = 2*(1+ beta**2)
        S = self.__S()
        print(np.max(np.abs(S))*self.hx**2)

        p_zero_xl = self.bound_cond.p_x_left_cond(self.y,self.t)
        
        print(p_zero_xl.shape)
        p_zero_yl = self.bound_cond.p_y_left_cond(self.x, self.t)
        print(p_zero_yl.shape)

        self.relax_steps = 10
        for _ in range(self.relax_steps):
            new_p_xr = self.__grid_left_right(new_p, type_is='p', direction= 'r', cond_axis = 'x')
            new_p_yr = self.__grid_left_right(new_p, type_is='p', direction= 'r', cond_axis = 'y')
            right_part = new_p + self.w/div *( new_p_xr + beta**2 * new_p_yr - self.hx**2 * S - div*new_p )

            for j in range(new_p.shape[0]):
                p_i_minus = p_zero_xl[j]
                P_j_minus = p_zero_yl if j==0 else new_p[j-1]
                for i in range(new_p.shape[1]):
                    p_j_minus = P_j_minus[i]
                    new_p[j,i] =right_part[j,i] + self.w/div* p_i_minus + self.w/div*beta**2 *p_j_minus
                    p_i_minus = new_p[j,i]
    
        self.p = new_p
        print('--------------')
    def update(self):
        
        print(self.u.shape, self.v.shape, self.p.shape)

        # D = self.__D()
        # print(D.shape)
        
        self.__update_p()
        print(self.p.shape)

        self._update_u()
        print(self.u.shape)

        self._update_v()
        print(self.v.shape)
      

        self.t += self.tau
        self.count +=1

if __name__ == "__main__":
    x_lim = [0,3.1]
    y_lim = [0,1]


    hx = 0.1
    hy = 0.1

    t_0 = 0
    tau = 0.1


    params = {
        "d" : y_lim[1],
        "rho": 1,
        "visc": 1
    }

    cond = BoundaryConditions_first_type(x_lim,y_lim,t_0,analytical_u,analytical_v,analytical_p,**params)

    x= np.arange(*x_lim, step =hx)[1:]
    y= np.arange(*y_lim, step =hy)[1:]
    xv, yv = np.meshgrid(x,y)
    
    m = Method(x,y,cond,hx,hy, tau, **params)


    print(absolute_error(analytical_u(xv,yv,t_0),m.Get_u()))
    print(absolute_error(analytical_v(xv,yv,t_0),m.Get_v()))
    print(absolute_error(analytical_p(xv,yv,t_0),m.Get_p()))

    t = t_0
    for i in range(1):
        m.update()
        t = t+ tau

    print(m.Get_u())

    print('________________________')
    print('u')
    print('absolute_error ', absolute_error(analytical_u(xv,yv,t),m.Get_u()))
    
    print('max value ',np.max(np.abs(analytical_u(xv,yv,t))))
    

    print('________________________')

    print('v')
    print('absolute_error ',absolute_error(analytical_v(xv,yv,t),m.Get_v()))
    print('max value ',np.max(np.abs(analytical_v(xv,yv,t))))


    print('________________________')
    print('p')
    print('absolute_error ',absolute_error(analytical_p(xv,yv,t),m.Get_p()))
    print('max value ',np.max(np.abs(analytical_p(xv,yv,t))))


    fig1, ax1 = plt.subplots()
    im1 = ax1.quiver(xv,yv, analytical_u(xv,yv,t), analytical_v(xv,yv,t),analytical_p(xv,yv,t), scale_units = "xy", scale = 50, angles ="xy")
    plt.colorbar(im1)
    # ax1.quiver(xv,yv, 0.000001*analytical_u(xv,yv,t), np.zeros(xv.shape))
    ax1.set_title('analytical')


    fig2, ax2 = plt.subplots()
    im2 =ax2.quiver(xv,yv, m.Get_u(),m.Get_v(),m.Get_p(),scale_units = "xy", scale = 50, angles ="xy")
    plt.colorbar(im2)
    ax2.set_title('test')
    
    plt.show()

    print(np.max(np.abs(analytical_p(xv,yv,0)))-np.min(np.abs(analytical_p(xv,yv,0))))
    print(np.min(analytical_p(xv,yv,0)))