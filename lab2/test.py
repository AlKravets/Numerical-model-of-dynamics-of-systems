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
    def __init__ (self, x, bound_cond, h, tau,**params):
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
        self.gamma = self._params.get('gamma') or 1
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

    def _x_2derivative(self):
        """
        Численная вторая производная по x.
        """
        # TODO
        pass
        # # Сдвинутая влево сетка (кравевое условие у меньшей границы)
        xgrid_l = np.hstack((self.bound_cond.x_left_cond(self.t),self.xgrid))[:-1]
        
        # Сдвинутая вправо сетка (кравевое условие у большей границы)
        xgrid_r = np.hstack((self.xgrid, self.bound_cond.x_right_cond(self.t)))[1:]
        return (xgrid_l - 2*self.xgrid + xgrid_r)/self.h**2
        
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

# не нужна
class ImplicitDifferenceScheme_Lite(ABC_Method):
    """
    Не полностью неявная разностная схема,
    """

    def __call__(self):
        return self.xgrid

    def update(self):
        # матрица для системы лин. алгебр. уравнений
        A = np.diagflat(-1*self.tau* self.beta*self.xgrid[1:]/2/h - self.tau*self.gamma/self.h**2 ,k=-1) +\
            np.diagflat(np.ones(self.xgrid.shape)* (1 + 2*self.tau*self.gamma/self.h**2),k=0) +\
                np.diagflat(self.tau* self.beta*self.xgrid[:-1]/2/h - self.tau*self.gamma/self.h**2 ,k=1)
        # правая часть системы лин. алгебр. уравнений
        F = np.copy(self.xgrid)
        F[0] += -1*self.bound_cond.x_left_cond(self.t+self.tau) *\
            (-1*self.tau *self.beta* self.xgrid[0]/2/h - self.tau*self.gamma/self.h**2)
        F[-1] += -1* self.bound_cond.x_right_cond(self.t+self.tau) *\
            (self.tau *self.beta*self.xgrid[-1]/2/h - self.tau*self.gamma/self.h**2)

        self.xgrid = np.linalg.solve(A,F)

        self.t += self.tau
        self.count +=1


class ImplicitDifferenceScheme(ABC_Method):

    def __call__(self):
        return self.xgrid

    def __u_left_right(self, u, t, direction = 'l'):
        '''
        direction = 'r' or 'l' 
        (right/left)
        '''
        if direction[0] == 'r':
            return np.hstack((u[1:],self.bound_cond.x_right_cond(t)))
        else:
            return np.hstack((self.bound_cond.x_left_cond(t), u[:-1]))


    def _newton_method(self, steps = 20):
        start = self.xgrid.copy()
        st_l = self.__u_left_right(start, self.t+self.tau, direction='l')
        st_r = self.__u_left_right(start, self.t+self.tau, direction='r')

        for i in range(steps):
            start = start - self.__jacobian_newton(start, st_l, st_r) @ self.__func_newton(start, st_l, st_r)
            st_l = self.__u_left_right(start, self.t+self.tau, direction='l')
            st_r = self.__u_left_right(start, self.t+self.tau, direction='r')
        
        return start

    
    def __func_newton(self, u, u_l, u_r):
        return u* (1 + 2*self.gamma * self.tau / self.h**2) + self.beta * self.tau /(2*self.h) * u *(u_r - u_l) - self.gamma * self.tau/ self.h**2 *(u_r+ u_l) - self.xgrid


    def __jacobian_newton(self, u, u_l, u_r):
        
        diag = (1+ 2*self.gamma * self.tau / self.h**2) + self.beta*self.tau/(2*self.h)*(u_r - u_l)
        diag_r =self.beta* self.tau/(2*self.h) * u[1:]  - self.gamma*self.tau/self.h**2
        diag_l= -self.beta*self.tau/(2*self.h) * u[:-1]  - self.gamma*self.tau/self.h**2

        Jacob = np.diagflat(diag, k=0) + np.diagflat(diag_l, k=-1) + np.diagflat(diag_r, k=1)

        return np.linalg.inv(Jacob)



    def update(self):
        self.xgrid = self._newton_method()

        self.t+=self.tau
        self.count +=1

class TwoStepFiniteDifferenseAlgorithm(ABC_Method):
    
    
    def __call__(self):
        print(self.xgrid.shape)
        return self.xgrid

    def __u_left_right(self, u, t, direction = 'l'):
        '''
        direction = 'r' or 'l' 
        (right/left)
        '''
        if direction[0] == 'r':
            return np.hstack((u[1:],self.bound_cond.x_right_cond(t)))
        else:
            return np.hstack((self.bound_cond.x_left_cond(t), u[:-1]))

    def __L(self,u, u_l, u_r):
        return -1*self.beta * u *(u_r - u_l)/(2* self.h) +\
                self.gamma *(u_r - 2*u + u_l)/self.h**2

    def update(self):
        u_r = self.__u_left_right(self.xgrid, self.t, direction='r')
        u_l= self.__u_left_right(self.xgrid, self.t, direction='l')

        new_xg_explis = self.xgrid + self.tau * self.__L(self.xgrid, u_l, u_r)

        new_xg_explis_r =self.__u_left_right(new_xg_explis, self.t + self.tau, direction='r')
        new_xg_explis_l =self.__u_left_right(new_xg_explis, self.t + self.tau, direction='l')

        new_xg_implis = (self.xgrid+  self.tau * self.gamma /self.h**2 * (new_xg_explis_r+ new_xg_explis_l))/ \
            (1 + self.tau*self.beta / (2* self.h) *(new_xg_explis_r - new_xg_explis_l) + self.tau*self.gamma *2 / self.h**2)
        
        for i in range(self.xgrid.shape[0]):
            if (i+ self.count) % 2 == 0:
                self.xgrid[i] = new_xg_explis[i]
            else: 
                self.xgrid[i] = new_xg_implis[i]


        self.t += self.tau
        self.count+=1





def absolute_error(analytical_decision, method_decision):
    """
    Принимает 2 сетки решений, возвращает число
    """
    return np.max(np.abs(analytical_decision - method_decision))


if __name__ == "__main__":
    x_lim = [0,101]
    h = 1

    steps = 150

    t_0 =0
    # tau= h**2/4
    tau= 1/3
    t = t_0

    cond = BoundaryConditions_first_type(x_lim, t_0, analytical_solution)

    x= np.arange(*x_lim, step =h)[1:]

    method = ImplicitDifferenceScheme(x,cond, h, tau)
    # method = TwoStepFiniteDifferenseAlgorithm(x,cond, h, tau)

    for i in range(steps):
        method.update()
        t+=tau
        print(i)
    err = absolute_error(analytical_solution(x,t), method())

    print('err:',err)


    plt.plot(x, analytical_solution(x,t), label ='real')
    plt.plot(x, method(), label = 'test')
    plt.legend()

    plt.show()