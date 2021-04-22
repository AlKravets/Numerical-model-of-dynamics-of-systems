from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl


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
        return self.u.copy()

    def Get_v(self):
        """
        Возвращает сетку составляющей скорости по y в текущий момент
        """
        return self.v.copy()

    def Get_p(self):
        """
        Возвращает сетку давления в текущий момент
        """
        return self.p.copy()

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
        

    def _update_p(self):
        # print('--------------')
        new_p = self.p.copy()

        beta = self.hx/self.hy

        _xi = ( ( np.cos(np.pi / (self.bound_cond.x_lim[1] -self.bound_cond.x_lim[0])/self.hx) + beta**2 *np.cos(np.pi / (self.bound_cond.y_lim[1] -self.bound_cond.y_lim[0])/self.hy)  )/(1 + beta**2))**2

        self.w = 2* ( 1 - np.sqrt(1 - _xi))/_xi
        
        # print(self.w)

        div = 2*(1+ beta**2)
        S = self.__S()
        # print(np.max(np.abs(S))*self.hx**2)

        p_zero_xl = self.bound_cond.p_x_left_cond(self.y,self.t)
        
        # print(p_zero_xl.shape)
        p_zero_yl = self.bound_cond.p_y_left_cond(self.x, self.t)
        # print(p_zero_yl.shape)

        
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
        # print('--------------')
    def update(self):
        """
        Обновление сетки (x,y), шаг по времени
        """
        
                
        self._update_p()
        

        self._update_u()
        

        self._update_v()
        
      

        self.t += self.tau
        self.count +=1



def main_and_error_plots(x_lim, y_lim, hx, hy, t_0, tau, steps, first_moment = 1, nsize = 1, cmap_name = 'viridis', **params):
    """
    Главная функция для создания отчета
    принимает ограничения по x: x_lim, по y: y_lim
    шаг по иксу: hx, шаг по игреку: hy
    начальное время: t_0
    шаг по времени: tau
    Число шагов, что надо сделать: steps
    число шагов для первой части рисунка: first_moment (советую брать малое число шагов, чтобы было видно скорости)
    nsize регулирует размер графиков
    cmap_name -- название тепловой карты. По ссылке можно выбрать другую
    https://matplotlib.org/stable/tutorials/colors/colormaps.html
    **params - это словарь константами для функций.
    """
    t = t_0

    cond = BoundaryConditions_first_type(x_lim,y_lim,t_0,analytical_u,analytical_v,analytical_p,**params)

    x= np.arange(*x_lim, step =hx)[1:]
    y= np.arange(*y_lim, step =hy)[1:]
    xv, yv = np.meshgrid(x,y)

    m = Method(x,y,cond,hx,hy, tau, **params)

    errors = [[],[],[]]

    first_steps =  first_moment if first_moment < steps else 1
    first_t = t_0

    for step in range(steps):
        print('step: ',step+1)
        m.update()
        t = t+ tau    

        errors[0].append(absolute_error(analytical_u(xv,yv,t),m.Get_u()))
        errors[1].append(absolute_error(analytical_v(xv,yv,t),m.Get_v()))
        errors[2].append(absolute_error(analytical_p(xv,yv,t),m.Get_p()))

        if step == first_steps:
            first_u = m.Get_u()
            first_v = m.Get_v()
            first_p = m.Get_p()
            first_t = t

    # рисунок решений

    my_scale = int(np.max(np.abs(analytical_u(xv,yv,t_0))) *10)

    fig1 , axes1 = plt.subplots(nrows = 2, ncols = 2, figsize = (nsize*14,nsize*9))

    axes1 = axes1.ravel()
    for ax in axes1:
        ax.set_ylabel('$y$')
        ax.set_xlabel('$x$')
    cmap = plt.get_cmap(cmap_name)
    im = axes1[0].contourf(xv,yv,analytical_p(xv,yv,first_t),cmap= cmap)
    axes1[0].quiver(xv,yv, analytical_u(xv,yv,first_t), analytical_v(xv,yv,first_t), scale_units = "xy", scale = my_scale, angles ="xy")
    axes1[0].set_title(f'Analytical decision. $t = {round(first_t,2)}$')

    im = axes1[1].contourf(xv,yv,first_p,cmap= cmap)
    axes1[1].quiver(xv,yv, first_u,first_v,scale_units = "xy", scale = my_scale, angles ="xy")
    axes1[1].set_title(f'Modeling decision. $t = {round(first_t,2)}$')    
    plt.colorbar(im,ax=axes1[:2],cmap= cmap)

   
    im = axes1[2].contourf(xv,yv,analytical_p(xv,yv,t),cmap= cmap)
    axes1[2].quiver(xv,yv, analytical_u(xv,yv,t), analytical_v(xv,yv,t), scale_units = "xy", scale = my_scale, angles ="xy")
    axes1[2].set_title(f'Analytical decision. $t = {round(t,2)}$')

    im= axes1[3].contourf(xv,yv,m.Get_p(),cmap= cmap)
    axes1[3].quiver(xv,yv, m.Get_u(),m.Get_v(),scale_units = "xy", scale = my_scale, angles ="xy")
    axes1[3].set_title(f'Modeling decision. $t = {round(t,2)}$')    
    plt.colorbar(im,ax=axes1[2:],cmap= cmap)    
        
    # рисунок ошибок

    fig2, axes2 = plt.subplots(ncols = 3, figsize = (nsize*16,nsize*5))

    axes2 = axes2.ravel()
    
    titles = ['$u$ errors', '$v$ errors','$p$ errors']
    for i in range(3):
        axes2[i].plot(np.arange(steps), errors[i])
        axes2[i].set_title(titles[i])
        axes2[i].set_ylabel('absolute error')
        axes2[i].set_xlabel('steps')

    axes2[0].plot(np.arange(steps), errors[0])
    axes2[1].plot(np.arange(steps), errors[1])
    axes2[2].plot(np.arange(steps), errors[2])

    plt.show()


def plot_for_errors_in_lab():
    """
    Не нужна в общем случае
    Рисует график, который демонстрирует, что лабораторная работа работает плохо
    """
    x_lim = [0,3.1]
    y_lim = [0,1]


    hx = 0.1
    hy = 0.1

    t_0 = 0
    tau = 0.0001


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
    m.update()
    t = t_0+ tau

    my_scale = int(np.max(np.abs(analytical_u(xv,yv,t_0))) *10)
    fig1, axes = plt.subplots(ncols=2, figsize =(14, 5))
    ax1, ax2 = axes.ravel()
    cmap = plt.get_cmap('viridis')
    im1 =ax1.contourf(xv,yv,analytical_p(xv,yv,t),cmap= cmap)
    ax1.quiver(xv,yv, analytical_u(xv,yv,t), analytical_v(xv,yv,t), scale_units = "xy", scale = my_scale, angles ="xy")
    fig1.colorbar(im1,cmap= cmap,ax= ax1)
    ax1.set_title(f'Analytical decision. $t = {t}$')
    ax1.set_ylabel('$y$')
    ax1.set_xlabel('$x$')


    im2 =ax2.contourf(xv,yv,m.Get_p(),cmap= cmap)
    ax2.quiver(xv,yv, m.Get_u(),m.Get_v(),scale_units = "xy", scale = my_scale, angles ="xy")
    fig1.colorbar(im2,ax= ax2, cmap = cmap)
    ax2.set_title(f'Modeling decision. $t = {t}$')
    ax2.set_ylabel('$y$')
    ax2.set_xlabel('$x$')
    plt.show()

def anim_plots(x_lim, y_lim, hx, hy, t_0, tau, steps,name_analit = 'analit_anim.gif',name_model = 'model_anim.gif', decimal_places= 2, figsize = None,**params):
    """
    функция для создания анимации
    создает 2 файла с анимацией аналитического решения и с анимацией модели.
    принимает ограничения по x: x_lim, по y: y_lim
    шаг по иксу: hx, шаг по игреку: hy
    начальное время: t_0
    шаг по времени: tau
    Число шагов, что надо сделать: steps
    name_analit, name_model -- названия для файлов анимации
    figsize -- можно менять размер рисунка.
    **params - это словарь константами для функций.
    """
    
    t = [t_0]
    count = [0]
    cond = BoundaryConditions_first_type(x_lim,y_lim,t_0,analytical_u,analytical_v,analytical_p,**params)

    x= np.arange(*x_lim, step =hx)[1:]
    y= np.arange(*y_lim, step =hy)[1:]
    xv, yv = np.meshgrid(x,y)

    m = Method(x,y,cond,hx,hy, tau, **params)

    fig = plt.figure(figsize=figsize)

    ax = plt.axes(xlim = x_lim, ylim  =y_lim)
    ax.set_ylabel('$y$')
    ax.set_xlabel('$x$')
    my_scale = int(np.max(np.abs(analytical_u(xv,yv,t_0))) *10)

    qv = ax.quiver(xv,yv,analytical_u(xv,yv,t[0]),analytical_v(xv,yv,t[0]),scale_units = "xy", scale = my_scale, angles ="xy")

    def init1():
        qv.set_UVC(analytical_u(xv,yv,t[0]),analytical_v(xv,yv,t[0]))
        ax.set_title(f'Analytical decision. steps: {count[0]}, time: {round(t[0], decimal_places)}')
        return qv,
    
    def animate1(i):
        t[0]+=tau
        count[0] +=1
        qv.set_UVC(analytical_u(xv,yv,t[0]),analytical_v(xv,yv,t[0]))
        ax.set_title(f'Analytical decision. steps: {count[0]}, time: {round(t[0], decimal_places)}')
        return qv,

    anim1 = animation.FuncAnimation(fig, animate1, init_func=init1,
                            frames=steps,interval=100, blit=True)
    anim1.save(name_analit)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(xlim = x_lim, ylim  =y_lim)
    ax.set_ylabel('$y$')
    ax.set_xlabel('$x$')
    my_scale = int(np.max(np.abs(analytical_u(xv,yv,t_0))) *10)

    qv = ax.quiver(xv,yv,m.Get_u(),m.Get_v(),scale_units = "xy", scale = my_scale, angles ="xy")
    def init2():
        qv.set_UVC(m.Get_u(),m.Get_v())
        ax.set_title(f'Modeling decision. steps: {m.count}, time: {round(m.t, decimal_places)}')
        return qv,
    
    def animate2(i):
        m.update()
        qv.set_UVC(m.Get_u(),m.Get_v())
        ax.set_title(f'Modeling decision. steps: {m.count}, time: {round(m.t, decimal_places)}')
        return qv,

    anim1 = animation.FuncAnimation(fig, animate2, init_func=init2,
                            frames=steps,interval=100, blit=True)
    anim1.save(name_model)



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


    # print('________________________')
    # print('u')
    # print('absolute_error ', absolute_error(analytical_u(xv,yv,t),m.Get_u()))
    
    # print('max value ',np.max(np.abs(analytical_u(xv,yv,t))))
    

    # print('________________________')

    # print('v')
    # print('absolute_error ',absolute_error(analytical_v(xv,yv,t),m.Get_v()))
    # print('max value ',np.max(np.abs(analytical_v(xv,yv,t))))


    # print('________________________')
    # print('p')
    # print('absolute_error ',absolute_error(analytical_p(xv,yv,t),m.Get_p()))
    # print('max value ',np.max(np.abs(analytical_p(xv,yv,t))))


    steps =100
    main_and_error_plots(x_lim,y_lim,hx,hy,t_0,tau,steps, **params)
    
    
    plot_for_errors_in_lab()

    steps = 20
    anim_plots(x_lim,y_lim,hx,hy,t_0,tau,steps, **params)
