from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt


def analytical_solution(x,t, **params):
    """
    аналитическое решение u*
    x,t : numpy array (or float)
    output: numpy array (or float)
    может принимать дополнительные параметры:
    u_01 = 3, (значения по умолчанию)
    u_02 = 1,
    gamma = 1
    """

    u_01 = params.get('u_01') or 3
    u_02 = params.get('u_02') or 1
    gamma = params.get('gamma') or 1

    c = (u_01 + u_02)/2

    return u_02 + (u_01 - u_02)/(1 + np.exp((u_01 -u_02)*(x - c*t)/2/gamma))


class ABC_BoundaryConditions_first_type(ABC):
    """
    Абстрактный класс для создания краевых условий первого рода
    """
    @abstractmethod
    def x_left_cond(self,t):
        """
        функция ограничения на левом крае по x
        """
        pass
    @abstractmethod
    def x_right_cond(self,t):
        """
        функция ограничения на правом крае по x
        """
        pass   
    @abstractmethod
    def time_cond(self, x):
        """
        функция ограничения в начальный момент по времени
        """
        pass

class ABC_Method(ABC):
    """
    Абстрактный класс методов, все наследуются от него.
    в нем перечислены обязательные методы
    """
    def __init__ (self, x, bound_cond, h, tau, right_part,**params):
        """
        bound_cond - это представитель класса HomogeneousBoundaryConditions_first_type
        right_part - функция, для правой части
        Важно заметить, что x подается как массив значений без краевых точек!!
        в Данном случае все краевые точки считаются из bound_cond
        в params записаны переменные u_01, u_02, gamma, beta
        beta = 1
        """
        self._params = params
        
        self.beta = self._params.get('beta') or 1
        self.x = x
                
        # краевые условия HomogeneousBoundaryConditions_first_type 
        self.bound_cond = bound_cond
        
        # Начальное значение времени
        self.t = self.bound_cond.t_0
        # Количество обновлений (шагов по времени)
        self.count = 0

        # Шаг по пространству
        self.h = h
        # Шаг по времени
        self.tau = tau


        
        self.xgrid = self.bound_cond.time_cond(self.x)

        self.right_part = right_part

    def _x_2derivative(self):
        """
        Численная вторая производная по x.
        """
        # TODO
        pass
        # # Сдвинутая влево сетка (v_k-1_m) (кравевое условие у меньшей границы)
        # xv_l = np.hstack((self.bound_cond.x_left_cond(self.y,self.t).reshape(-1,1),self.xy))[:,:-1]
        
        # # print(xv_l.shape)
        # # Сдвинутая вправо сетка (v_k+1_m) (кравевое условие у большей границы)
        # xv_r = np.hstack([self.xy, self.bound_cond.x_right_cond(self.y, self.t).reshape(-1,1)])[:,1:]
        # # print(xv_r.shape)
        # # print(xv_r)
        # # print(self.xy.shape)
        # return (xv_l - 2* self.xy + xv_r)/self.h**2


    @abstractmethod
    def __call__(self):
        """
        Должен возвращать значение сетки (x, y) в текущий момент
        """
        pass
    @abstractmethod
    def update(self):
        """
        Обновление сетки (x,y), шаг по времени
        """
        pass


class BoundaryConditions_first_type(ABC_BoundaryConditions_first_type):
    """
    Кравевые условия первого рода считаются для 1 варианта
    """
    def __init__(self,x_lim, t_0, analytical_solution, **params):
        self.x_lim = x_lim # [x_min, x_max]
        self.t_0 = t_0
        # функция аналитического решения
        self._analytical_solution = analytical_solution
        # параметры для аналитического решения
        self._params = params

    def x_left_cond(self,t):
        return self._analytical_solution(self.x_lim[0], t, **self._params)
    
    def x_right_cond(self,t):
        return self._analytical_solution(self.x_lim[1], t, **self._params)
    def time_cond(self,x):
        return self._analytical_solution(x, self.t_0, **self._params)


class ImplicitDifferenceScheme(ABC_Method):
    """
    Неявная разностная схема
    """

    def __call__(self):
        return self.xgrid

    def update(self):
        pass



if __name__ == "__main__":
    n = 3
    a = np.zeros((n,n))
    b = np.arange(5,5+n-1)
    a = a+ 2*np.diagflat(np.ones(n), k=0) + 3*np.diagflat(np.ones(n-1), k=-1) + np.diagflat(b, k=1)
    

    print(a)
