from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

def analytical_solution_1(x,y,t, **params):
    """
    Это аналитаческое решение (w) для вар 1
    в функцию по своему желанию можно передавать именованные параметры 
    a="value",
    A="value",
    k1="value",
    k2="value"
    если ничего не передано,то идет значения по умолчанию: все равны 1
    """
    a = params.get("a") or 1
    A = params.get("A") or 1
    k1 = params.get("k1") or 1
    k2 = params.get("k2") or 1
    return A*np.exp(k1*x + k2*y + (k1**2 + k2**2)*a*t)

def analytical_f_1(x,y,t,**params):
    """
    Аналитически вычисленная правая часть по аналитическому решению
    В случае вар. 1 это просто 0
    """
    return np.zeros(x.shape)


class ABC_Method(ABC):
    """
    Абстрактный класс методов, все наследуются от него.
    в нем перечислены обязательные методы
    """
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
    
class ABC_BoundaryСonditions_first_type(ABC):
    """
    Абстрактный класс для создания кравевых условий первого рода
    """
    @abstractmethod
    def x_left_cond(self,y,t):
        """
        функция ограничения на левом крае по x
        """
        pass
    @abstractmethod
    def x_right_cond(self, y,t):
        """
        функция ограничения на правом крае по x
        """
        pass
    @abstractmethod
    def y_left_cond(self,x,t):
        """
        функция ограничения на левом крае по y
        """
        pass
    @abstractmethod
    def y_right_cond(self, x,t):
        """
        функция ограничения на правом крае по y
        """
        pass
    @abstractmethod
    def time_cond(self, x,y):
        """
        функция ограничения в начальный момент по времени
        """

# Что-то тут не так
class HomogeneousBoundaryConditions_first_type(ABC_BoundaryСonditions_first_type):
    """
    класс однородных граничных условий первого рода
    все условия нулевые
    """
    def __init__(self,x_lim, y_lim, t_0):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.t_0 = t_0
    def x_left_cond(self, y, t):
        return np.zeros(y.shape)
    def x_right_cond(self, y, t):
        return np.zeros(y.shape)
    def y_left_cond(self, x, t):
        return np.zeros(x.shape)
    def y_right_cond(self, x, t):
        return np.zeros(x.shape)
    def time_cond(self,x,y):
        return np.zeros(x.shape)


class BoundaryConditions_first_type(ABC_BoundaryСonditions_first_type):
    """
    Кравевые условия первого рода считаются для 1 варианта
    """
    def __init__(self,x_lim, y_lim, t_0):
        self.x_lim = x_lim # [x_min, x_max]
        self.y_lim = y_lim
        self.t_0 = t_0
    def x_left_cond(self, y, t):
        _x = np.ones(y.shape)* self.x_lim[0]
        return analytical_solution_1(_x,y,t)
    def x_right_cond(self, y, t):
        _x = np.ones(y.shape)* self.x_lim[1]
        return analytical_solution_1(_x,y,t)
    def y_left_cond(self, x, t):
        _y = np.ones(x.shape)* self.y_lim[0]
        return analytical_solution_1(x,_y,t)
    def y_right_cond(self, x, t):
        _y = np.ones(x.shape)* self.y_lim[1]
        return analytical_solution_1(x,_y,t)
    def time_cond(self,x,y):
        xv, yv = np.meshgrid(x,y)
        return analytical_solution_1(xv,yv,self.t_0)




class MethodExplicitDifferenceScheme(ABC_Method):
    """
    Явная разностная схема с однородными кравевыми условиями
    """
    def __init__ (self, x, y, bound_cond, h, tau, right_part,**params):
        """
        bound_cond - это представитель класса HomogeneousBoundaryConditions_first_type
        right_part - функция, для правой части
        Важно заметить, что x,y подаются как массивы значений без краевых точек!!
        в Данном случае все краевые точки считаются из bound_cond
        в params записаны переменные a, A, k1, k2
        a= "value"
        """
        self._params = params

        self.x = x
        self.y = y
        
        # краевые условия HomogeneousBoundaryConditions_first_type 
        self.bound_cond = bound_cond
        
        # Начальное значение времени
        self.t = self.bound_cond.t_0
        # Количесво обновлений (шагов по времени)
        self.count = 0

        # Шаг по пространству
        self.h = h
        # Шаг по времени
        self.tau = tau

        self._xv, self._yv = np.meshgrid(self.x,self.y)
        # состояние сетки в данный момент
        # на начальном шаге инициализируем из кравевых условий
        self.xy = self.bound_cond.time_cond(self.x,self.y)

        self.right_part = right_part

    def __call__(self):
        return self.xy

    def _x_2derivative(self):
        """
        Численная вторая производная по x.
        """
        # Сдвинутая влево сетка (v_k-1_m) (кравевое условие у меньшей границы)
        xv_l = np.hstack((self.bound_cond.x_left_cond(self.y,self.t).reshape(-1,1),self.xy))[:,:-1]
        
        # print(xv_l.shape)
        # Сдвинутая вправо сетка (v_k+1_m) (кравевое условие у большей границы)
        xv_r = np.hstack([self.xy, self.bound_cond.x_right_cond(self.y, self.t).reshape(-1,1)])[:,1:]
        # print(xv_r.shape)
        # print(xv_r)
        # print(self.xy.shape)
        return (xv_l - 2* self.xy + xv_r)/self.h**2
    def _y_2derivative(self):
        """
        Численная вторая производная по y.
        """
        # Сдвинутая вниз сетка (v_k_m-1) (кравевое условие у меньшей границы)
        yv_l = np.vstack([self.bound_cond.y_left_cond(self.x,self.t).reshape(1,-1), self.xy])[:-1,:]
        # Сдвинутая вверх сетка (v_k_m+1) (кравевое условие у большей границы)
        yv_r = np.vstack([self.xy, self.bound_cond.y_right_cond(self.x, self.t).reshape(1,-1)])[1:,:]

        

        return (yv_l - 2* self.xy + yv_r)/self.h**2
    
    def update(self):
        a = self._params.get('a') or 1
        self.xy = self.tau*( self.right_part(self._xv,self._yv, self.t, **self._params) + a*self._x_2derivative() + a*self._y_2derivative()) + self.xy

        self.t+= self.tau
        self.count +=1




if __name__ == "__main__":
    # x = np.linspace(1,2,4)
    # x1 = np.array((0))
    # y = np.linspace(2, 5, 5)


    # print(x.shape, x1.shape, y.shape)

    # xv,yv= np.meshgrid(x,y)
    # print(xv.shape, yv.shape)

    # x1v, y1v = np.meshgrid(x1,y)
    # print(x1v.shape, y1v.shape)

    # r = np.hstack((xv,x1v))
    # print(r.shape)
    # print(r)

    # r = np.hstack((x1v,xv))
    # print(r.shape)
    # print(r)

    x_lim = [0,1]
    y_lim = [0,1]
    h= 0.01
    tau = h**2/4
    t_0= 0

    cond= BoundaryConditions_first_type(x_lim,y_lim,t_0)

    x= np.arange(*x_lim, step =h)[1:]
    y = np.arange(*y_lim, step =h)[1:]
    
    method = MethodExplicitDifferenceScheme(x,y,cond, h, tau, analytical_f_1)

    for i in range(10000):
        method.update()
        print(i)

    xv, yv = np.meshgrid(x,y)
    print(np.max(method() - analytical_solution_1(xv,yv,method.t)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(xv,yv,analytical_solution_1(xv,yv,method.t))
    surf2 = ax.plot_surface(xv,yv,method())
    plt.show()


    # print(method._x_2derivative())
    # print(analytical_solution_1(x[-1], y[-1], method.t))

    print(cond.x_right_cond(y, t_0))


