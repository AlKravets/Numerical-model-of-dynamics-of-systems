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
    a = params.get("a") or 1
    A = params.get("A") or 1
    k1 = params.get("k1") or 1
    k2 = params.get("k2") or 1
    return np.zeros(x.shape)


class ABC_Method(ABC):
    """
    Абстрактный класс методов, все наследуются от него.
    в нем перечислены обязательные методы
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
    def __init__(self,x_lim, y_lim, t_0, analytical_solution, **params):
        self.x_lim = x_lim # [x_min, x_max]
        self.y_lim = y_lim
        self.t_0 = t_0
        # функция аналитического решения
        self._analytical_solution = analytical_solution
        # параметры для аналитического решения
        self._params = params

    def x_left_cond(self, y, t):
        _x = np.ones(y.shape)* self.x_lim[0]
        return self._analytical_solution(_x,y,t,**self._params)
    def x_right_cond(self, y, t):
        _x = np.ones(y.shape)* self.x_lim[1]
        return self._analytical_solution(_x,y,t,**self._params)
    def y_left_cond(self, x, t):
        _y = np.ones(x.shape)* self.y_lim[0]
        return self._analytical_solution(x,_y,t,**self._params)
    def y_right_cond(self, x, t):
        _y = np.ones(x.shape)* self.y_lim[1]
        return self._analytical_solution(x,_y,t,**self._params)
    def time_cond(self,x,y):
        xv, yv = np.meshgrid(x,y)
        return self._analytical_solution(xv,yv,self.t_0, **self._params)




class MethodExplicitDifferenceScheme(ABC_Method):
    """
    Явная разностная схема с однородными кравевыми условиями
    """
    ## init определен в родителе

    def __call__(self):
        return self.xy

    
    def update(self):
        a = self._params.get('a') or 1
        self.xy = self.tau*( self.right_part(self._xv,self._yv, self.t, **self._params) + a*self._x_2derivative() + a*self._y_2derivative()) + self.xy

        self.t+= self.tau
        self.count +=1


class AlternatingDirectionMethod(ABC_Method):
    """
    Метод переменных направлений (дробных шагов)
    """
    def __init__(self, x, y, bound_cond, h, tau, right_part,**params):
        # инициализация как в родительском классе
        super().__init__(x, y, bound_cond, h, tau, right_part,**params)
        
        
        # параметр a
        self._a = self._params.get('a') or 1


        # матрица для метода прогонки        
        self.__matrix = np.zeros((self.y.shape[0], self.x.shape[0]))
        self.__matrix[0][0], self.__matrix[0][1] = -1*(1+self._a*self.tau/self.h**2), self._a*self.tau/(self.h**2 *2)
        self.__matrix[-1][-2], self.__matrix[-1][-1] =  self._a*self.tau/(self.h**2 *2), -1*(1+self._a*self.tau/self.h**2)

        for i in range(1,self.__matrix.shape[0]-1):
            self.__matrix[i][i] = -1*(1+self._a*self.tau/self.h**2)
            self.__matrix[i][i-1] = self._a*self.tau/(self.h**2 *2)
            self.__matrix[i][i+1] = self._a*self.tau/(self.h**2 *2)

        # print(self.__matrix)



    def __call__(self):
        return self.xy

    def _first_half_step(self):
        """
        Первая часть обновления, изменяет xy
        Шаг по координате x
        но из-за meshgrid, х - это вторая координата в массиве arr[y,x]
        """

        new_xy = []
        ## Правая часть для метода прогонки
        F = self.tau/2 * (-1*self._a*self._y_2derivative() - self.right_part(self._xv, self._yv, self.t+ self.tau/2, **self._params)) - self.xy
        F[:,0] = F[:,0] - self._a*self.tau/(2* self.h**2)* self.bound_cond.x_left_cond(self.y, self.t+ self.tau/2)
        F[:,-1] = F[:,-1] - self._a*self.tau/(2* self.h**2)* self.bound_cond.x_right_cond(self.y, self.t+ self.tau/2)

        # Тут нужен метод прогонки, но я ленивый и использовал втроенную функцию numpy
        for f in F:
            new_xy.append(np.linalg.solve(self.__matrix, f))

        self.xy = np.array(new_xy)
        ## увеличили время
        self.t += self.tau/2
    
    def _second_half_step(self):
        """
        Вторая часть обновления, изменяет xy
        но из-за meshgrid, y - это первая координата в массиве arr[y,x]
        """
        new_xy = []
        ## Правая часть для метода прогонки
        F = self.tau/2 * (-1*self._a*self._x_2derivative() - self.right_part(self._xv, self._yv, self.t, **self._params)) - self.xy
        F[0] = F[0] - self._a*self.tau/(2* self.h**2)* self.bound_cond.y_left_cond(self.x, self.t+ self.tau/2)
        F[-1] = F[-1] - self._a*self.tau/(2* self.h**2)* self.bound_cond.y_right_cond(self.y, self.t+ self.tau/2)

        # Тут нужен метод прогонки, но я ленивый и использовал втроенную функцию numpy
        for f in F.T:
            new_xy.append(np.linalg.solve(self.__matrix, f))

        self.xy = np.array(new_xy).T
        ## увеличили время
        self.t += self.tau/2

    def update(self):
        self._first_half_step()
        # print(self.xy.shape)
        self._second_half_step()
        # print(self.xy.shape)

        self.count+=1


def absolute_error(analytical_decision, method_decision):
    """
    Принимает 2 сетки решений, возвращает число
    """
    return np.max(np.abs(analytical_decision - method_decision))

def error_graphic(t_list, methods_errors_list, methods_labels = None , step =1, title = "", log_scale= False):
    """
    t_list: список с временными отметками
    methods_errors: список списков с ошибками (по первой координате - методы, по второй - ошибки в соответствующий момент времени)
            [[method1_err1, ...], [method2_err1, ...],... ]
    step = 1 пропуск времени (если поставить  step =10  то выведется только каждый 10 результат)
    """

    m_labels = methods_labels or [f'method {i}' for i in range(len(methods_errors_list))]

    fig, ax = plt.subplots()
    
    ax.set_title(title)
    # ax.set_xlim()
    # ax.set_ylim()
    ax.set_xlabel("$t$")
    ax.set_ylabel("absolute error")
    if log_scale:
        ax.set_yscale('log')
    for i in range(len(methods_errors_list)):
        ax.plot(t_list[::step], methods_errors_list[i][::step], label = m_labels[i])
    ax.legend()
    return fig

def solution_graphic (xv, yv, results, res_lables = None, title = ''):
    """
    График решения. xv, yv - сетка (получили через meshgrid)
    results - список результатов аналитической функции и методов
    """
    
    labels = res_lables or [f'result {i}' for i in range(len(res_lables))]

    fig, ax = plt.subplots()
    
    ax.set_title(title)
    # ax.set_xlim()
    # ax.set_ylim()
    
    for i in range(len(results)):
        ax.plot_surface(xv,yv, results[i], label = labels[i])
    ax.legend()
    return fig


class Double_strandedSymmetrizedAlgorithm(ABC_Method):
    """
    Явная разностная схема с однородными кравевыми условиями
    """
    ## init определен в родителе

    def __call__(self):
        return self.xy

    
    def update(self):
        a = self._params.get('a') or 1
        # обновление по явной схеме (это должно происходить только на парных точках, но обновить все проще)
        explisit_xy = self.tau*( self.right_part(self._xv,self._yv, self.t, **self._params) + a*self._x_2derivative() + a*self._y_2derivative()) + self.xy



        ## Вычисляем части производных для неявного шага метода
        #Сдвинутая влево сетка (v_k-1_m) (кравевое условие у меньшей границы)
        xv_l = np.hstack((self.bound_cond.x_left_cond(self.y,self.t+self.tau).reshape(-1,1),explisit_xy))[:,:-1]
        
        # Сдвинутая вправо сетка (v_k+1_m) (кравевое условие у большей границы)
        xv_r = np.hstack([explisit_xy, self.bound_cond.x_right_cond(self.y, self.t+self.tau).reshape(-1,1)])[:,1:]

        # Сдвинутая вниз сетка (v_k_m-1) (кравевое условие у меньшей границы)
        yv_l = np.vstack([self.bound_cond.y_left_cond(self.x,self.t+self.tau).reshape(1,-1), explisit_xy])[:-1,:]
        # Сдвинутая вверх сетка (v_k_m+1) (кравевое условие у большей границы)
        yv_r = np.vstack([explisit_xy, self.bound_cond.y_right_cond(self.x, self.t+self.tau).reshape(1,-1)])[1:,:]

        # Обовление по неявной схеме
        non_explisit_xy = ( self.right_part(self._xv, self._yv, self.t+self.tau, **self._params) + self.xy/self.tau + a*(xv_l + xv_r)/self.h**2 + a*(yv_l + yv_r)/self.h**2 ) *\
                            self.tau * self.h**2 / (self.h**2 + 4*a*self.tau)

        
        # совмещение(отбор четных и нечетных элементов) явного и неявного обновления
        for i in range(self.xy.shape[0]):
            for j in range(self.xy.shape[1]):
                if i+j+self.count % 2 ==0:
                    self.xy[i,j] = explisit_xy[i,j]
                else:
                    self.xy[i,j] = non_explisit_xy[i,j]



        self.t += self.tau
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
    h= 0.1
    tau = h**2/4
    t_0= 0

    params = {
        'a':0.01
    }

    cond= BoundaryConditions_first_type(x_lim,y_lim,t_0, analytical_solution_1, **params)

    x= np.arange(*x_lim, step =h)[1:]
    y = np.arange(*y_lim, step =h)[1:]
    method1 = MethodExplicitDifferenceScheme(x,y,cond, h, tau, analytical_f_1, **params )
    # method2 = Double_strandedSymmetrizedAlgorithm(x,y,cond, h, tau, analytical_f_1, **params)
    method2 = AlternatingDirectionMethod(x,y,cond, h, tau, analytical_f_1, **params)

    t_l = []
    m_e_l = [[],[]]

    xv, yv = np.meshgrid(x,y)

    for i in range(100):
        t_l.append(method1.t)
        # t_l.append(method2.t)
        m_e_l[0].append(absolute_error(analytical_solution_1(xv,yv,method1.t, **params), method1()))
        m_e_l[1].append(absolute_error(analytical_solution_1(xv,yv,method2.t, **params), method2()))


        method1.update()
        method2.update()
        print(i)

    print(m_e_l[0][:10])
    error_graphic(t_l, m_e_l, title= "test")
    
    print(np.max(np.abs(method1() - analytical_solution_1(xv,yv,method1.t, **params))))
    print(np.max(np.abs(method2() - analytical_solution_1(xv,yv,method2.t, **params))))
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(xv,yv,analytical_solution_1(xv,yv,method1.t, **params))
    surf2 = ax.plot_surface(xv,yv,method1())
    # print(method.t)
    plt.show()

    # print(method.xy[-1])
    # print(analytical_solution_1(x[-1], y[-1], method.t))

    # print(cond.x_right_cond(y, t_0))
    


