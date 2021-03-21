from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



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
    """
    Неявная разностная схема.
    """
    
    def __init__(self,*args,newton_lite = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._newton_lite = newton_lite



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

        if not self._newton_lite:
            for i in range(steps):
                start = start - self.__jacobian_newton(start, st_l, st_r) @ self.__func_newton(start, st_l, st_r)
                st_l = self.__u_left_right(start, self.t+self.tau, direction='l')
                st_r = self.__u_left_right(start, self.t+self.tau, direction='r')
        else:
            print('!')
            Jacob_0 = self.__jacobian_newton(start, st_l, st_r)
            for i in range(steps):
                start = start - Jacob_0 @ self.__func_newton(start, st_l, st_r)
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

class TwoStepFiniteDifferenceAlgorithm(ABC_Method):
    """
    Двухшаговый симетризированный метод
    """
    
    def __call__(self):
        # print(self.xgrid.shape)
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



def main_plots_and_errors(x_lim, h, t_0, tau, steps_list, nsize = 1, delimiter = ' & ',decimal_places= 4,plots_names =None, **params):
    """
    Главная функция для создания отчета
    принимает ограничения по x x_lim
    шаг по иксу h
    начальное время t_0
    шаг по времени tau
    список с числами шагов, на которых нужно вывести график
    nsize регулирует размер графика
    delimiter, decimal_places отвечают за параметры таблицы ошибок
    **params - это словарь константами для функций.
    plots_names список что содержит подписи для [
        аналитического решения,
        метода ImplicitDifferenceScheme,
        метода TwoStepFiniteDifferenceAlgorithm
    ] 

    Возвращает график с 2 колонками картинок, строки картинок регулируются количеством элементов step_list
    Возвращает таблицу с количеством шагов и ошибками методов на этом количестве шагов.
    """
    
    t = t_0
    x= np.arange(*x_lim, step =h)[1:]

    cond = BoundaryConditions_first_type(x_lim, t_0, analytical_solution,  **params)

    method_im = ImplicitDifferenceScheme(x,cond, h, tau, **params)
    method_two_step = TwoStepFiniteDifferenceAlgorithm(x,cond, h, tau, **params)

    err_list = [[],[]]

    steps_list.sort()
    n = len(steps_list)

    if plots_names is None:
        plots_names = ['Analit. solution','Implicit scheme', 'Two steps sim. method']
    fig, ax = plt.subplots(n//2 +n%2,2, figsize = (nsize*15, nsize*15 *(n//2+n%2)/2) )
    ax = ax.ravel()

    for i in range(steps_list[-1]):
        print(f'step {i+1}')
        method_im.update()
        method_two_step.update()
        t+=tau
        if i+1 in steps_list:
            j = steps_list.index(i+1)
            ax[j].plot(x, analytical_solution(x,t, **params), label=plots_names[0])
            ax[j].plot(x, method_im(), label = plots_names[1])
            ax[j].plot(x, method_two_step(), label = plots_names[2])

            ax[j].legend()
            ax[j].set_title(f'steps: {method_im.count}, time: {round(method_im.t, decimal_places//2)}')

            err_list[0].append(absolute_error(analytical_solution(x,t, **params), method_im()))
            err_list[1].append(absolute_error(analytical_solution(x,t, **params), method_two_step()))
    
    for i in range(j+1, len(ax)):
        ax[i].remove()

    print('Error table')
    print("N", delimiter, delimiter.join([str(item) for item in steps_list]))
    for index, err in enumerate(err_list):
        print(plots_names[index+1], delimiter, delimiter.join([str(round(item,decimal_places)) for item in err]), sep='')

    plt.show()


def anim_plots(x_lim,h, t_0, tau, step, y_lim = (-0.5, 3.5), name = 'res_animation.gif', figsize = None, decimal_places= 2,plots_names =None, **params):
    """
    функция для создания анимации
    принимает ограничения по x x_lim
    Также ВАЖНО указать верные ограничения на y y_lim (по умолчанию они настроены для тестовой анимации)
    шаг по иксу h
    начальное время t_0
    шаг по времени tau
    Количество шагов step это также количество кадров анимации
    nsize регулирует размер графика
    delimiter, decimal_places отвечают за параметры таблицы ошибок
    **params - это словарь константами для функций.
    plots_names список что содержит подписи для [
        аналитического решения,
        метода ImplicitDifferenceScheme,
        метода TwoStepFiniteDifferenceAlgorithm
    ] 

    Возвращает график с 2 колонками картинок, строки картинок регулируются количеством элементов step_list
    Возвращает таблицу с количеством шагов и ошибками методов на этом количестве шагов.
    """
    t = [t_0]
    x= np.arange(*x_lim, step =h)[1:]

    cond = BoundaryConditions_first_type(x_lim, t_0, analytical_solution,  **params)

    method_im = ImplicitDifferenceScheme(x,cond, h, tau, **params)
    method_two_step = TwoStepFiniteDifferenceAlgorithm(x,cond, h, tau, **params)

    
    methdos_list = [analytical_solution, method_im, method_two_step]
    if plots_names is None:
        plots_names = ['Analit. solution','Implicit scheme', 'Two steps sim. method']

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(xlim = x_lim, ylim = y_lim)
    
    line_list = []
    for i in range(3):
        line, = ax.plot([],[], label = plots_names[i])
        line_list.append(line)
    
    def init():
        for index, line in enumerate(line_list):
            if index ==0:
                line.set_data(x, methdos_list[0](x, t[0], **params))
            else:
                line.set_data(x, methdos_list[index]())
        return line_list

    def animate(i):
        t[0] +=tau
        methdos_list[1].update()
        methdos_list[2].update()
        ax.set_title(f'steps: {method_im.count}, time: {round(method_im.t, decimal_places)}')
        for index, line in enumerate(line_list):
            if index ==0:
                line.set_data(x, methdos_list[0](x,t[0],**params))
            else:
                line.set_data(x,methdos_list[index]())

        return line_list
    plt.legend(framealpha = 0.4,loc=3)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=step,interval=100, blit=True)
    
    anim.save(name)
    # если выдает ошибки попробуйте
    # anim.save(name, writer='pillow')


if __name__ == "__main__":
    x_lim = [0,101]
    h = 1

    t_0 =0
    tau= 1/3

    params ={
        'u_01': 3,
        'u_02': 1,
        'gamma': 1,
        'beta': 1,
    }

    steps_list = [10,50,100,150]
    plots_names = ['Analit. solution','Implicit scheme', 'Two steps sim. method']

    main_plots_and_errors(x_lim, h, t_0, tau, steps_list, **params, plots_names= plots_names)

    step = 160

    # anim_plots(x_lim, h, t_0, tau, step, **params, figsize=(15,10), y_lim=(0,3.5), name= 'gif/res_animation.gif')