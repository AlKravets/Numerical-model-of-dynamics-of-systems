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
    def __init__(self, x, y, bound_cond, hx, hy, tau, w = 1.5, **params):
        """
        bound_cond - это представитель класса BoundaryConditions_first_type
        Важно заметить, что x,y подается как массив значений без краевых точек!!
        в Данном случае все краевые точки считаются из bound_cond
        w = 1.5 параметр релаксации 1 <= w <=2
        в params записаны переменные 
        rho, d
        """
        self.x = x
        self.y = y

        self._params = params
        self.w = w

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

    def __grid_left_right(self,type_is , direction, cond_axis):
        """
        type is = 'p' or 'v' or 'u'
        direction = 'r' or 'l' 
        (right/left)
        cond_axis = 'x' or 'y'
        """

        if type_is =='v':
            grid = self.v.copy()
        elif type_is == 'u':
            grid= self.u.copy()
        else:
            grid = self.p.copy()

        
        if cond_axis == 'x':
            if direction == 'l':
                grid = np.hstack([self.bound_cond.cond(self.x,self.y, self.t,direction, type_is, cond_axis).reshape(-1.1),
                    grid])[:,:-1]
            else:
                grid = np.hstack([grid, self.bound_cond.cond(self.x,self.y, self.t,direction, type_is, cond_axis).reshape(-1.1)])[:,1:]
        
        else:
            if direction == 'l':
                grid = np.hstack([self.bound_cond.cond(self.x,self.y, self.t,direction, type_is, cond_axis).reshape(-1.1), grid])[:-1,:]
            else:
                grid = np.hstack([grid, self.bound_cond.cond(self.x,self.y, self.t,direction, type_is, cond_axis).reshape(-1.1)])[1:,:]

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
        D = (self.__grid_left_right('u',direction= 'r', cond_axis= 'x') - 
                self.__grid_left_right('u',direction= 'l', cond_axis= 'x'))/(2*self.hx)  + \
            (self.__grid_left_right('v',direction='r',cond_axis= 'y') - 
                self.__grid_left_right('v',direction='l',cond_axis='y'))/(2*self.hy)

        return D

    def __S(self):
        """
        S из описания лабораторной работы (правая часть при вычислении давления)
        """
        S = self.__D()/self.tau
        
        # тут вопросы
        D = self.__D()
        S = S + 1/self.__RE() * (self.__D_left_right( direction='r',cond_axis='x') - 2*D +  
                                self.__D_left_right(direction='l', cond_axis='x'))/self.hx**2 + \
                            (self.__D_left_right(direction='r',cond_axis='y') - 2*D +
                            self.__D_left_right(direction='l', cond_axis= 'y'))/self.hy**2

        return S

    
    def update(self):
        pass
        

if __name__ == "__main__":
    x_lim = [0,1]
    y_lim = [0,1]

    hx = 0.1
    hy = 0.1
    tau = 0.01
