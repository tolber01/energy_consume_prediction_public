import pandas as pd
import numpy as np
from scipy.optimize import minimize


class WintersNoTrend:
    def __init__(self, s: int):
        self.s = s # период сезонности
        self.a = None
        self.theta = None
        self.n = None
    
    def fit(self, train: pd.DataFrame, alpha: float):
        y = train.iloc[:, 1]
        self.n = train.shape[0]
    
        self.a = np.zeros(self.n)
        self.theta = np.zeros(self.n)
        m = y.mean()
        
        for t in range(self.s):
            self.theta[t] = y[t]
            self.a[t] = m
        
        for t in range(self.s, self.n):
            if self.theta[t - self.s] != 0:
                self.a[t] = alpha[0] * (y[t] / self.theta[t - self.s]) \
                    + (1 - alpha[0]) * self.a[t - 1]
            else:
                self.a[t] = (1 - alpha[0]) * self.a[t - 1]
            if self.a[t] != 0:
                self.theta[t] = alpha[1] * (y[t] / self.a[t]) \
                    + (1 - alpha[1]) * self.theta[t - self.s]
            else:
                self.theta[t] = (1 - alpha[1]) * self.theta[t - self.s]
    
    def predict(self, d: int=1) -> np.array:
        y_predicted = np.zeros(d)
        
        for t in range(self.n, self.n + d):
            y_predicted[t - self.n] = self.a[t - d] \
                * self.theta[t - d + (d % self.s) - self.s]
        
        return y_predicted


def alpha_optimizer(alpha: float, s: int, train_data: pd.DataFrame, \
                    real_data: pd.DataFrame) -> float:
    winters_model = WintersNoTrend(s)
    winters_model.fit(train_data, alpha)
    predicted_data = winters_model.predict(real_data.shape[0])
    
    return ((predicted_data - real_data.iloc[:, 1].values)**2).sum()


def full_winters_pipeline(train_data: pd.DataFrame) -> pd.DataFrame:
    """ Функция, содержащая весь процесс получения прогноза 
    потребления от входных данных до результата прогнозирования.
    Принимает единственный аргумент train_data типа pd.DataFrame, 
    содержащий два столбца:
        - Первый - дата и время снятия потребления в формате 
            pandas.Timestamp или datetime.datetime;
        - Второй - собственно значение потребления.
    Предполагается, что в тренировочных данных предоставлены почасовые данные.
    Используется адаптивная модель Уинтерса без тренда с оптимизацией параметров 
    сглаживания на тренировочных данных.
    Определение количества прогнозируемых значений (от конца тренировочных данных 
    до конца последнего месяца) происходит автоматически.
    Функция возвращает результат прогноза в виде объекта pandas.DataFrame, 
    имеющего ту же структуру, что и train_data.
    """
    
    time_delta = train_data.iloc[1, 0] - train_data.iloc[0, 0]
    last_date = train_data.iloc[-1, 0]
    
    before_dates = [last_date]
    while (last_date - before_dates[-1]).days < 30:
        before_dates.append(before_dates[-1] - time_delta)
    
    data_slice = train_data.iloc[:, 0].isin(before_dates)
    train, real = train_data[~data_slice], train_data[data_slice]
    
    s = 7 * 24 # период сезонности (7 дней по 24 часа)
    alpha_optimized = minimize(lambda x: alpha_optimizer(x, s, train, real),
        x0=np.array([0, 1]),
        bounds=[(0, 1), (0, 1)]
    ).x # оптимизация параметров сглаживания в модели Уинтерса
    
    winters_model = WintersNoTrend(s)
    winters_model.fit(train_data, alpha_optimized)
    
    predict_dates = [last_date + time_delta] # даты, на которые будет выполняться прогноз
    while (predict_dates[-1] + time_delta).month == last_date.month:
        predict_dates.append(predict_dates[-1] + time_delta)
    
    d = len(predict_dates) # горизонт прогнозирования
    result = pd.DataFrame(data={
        train_data.columns[0]: predict_dates,
        train_data.columns[1]: winters_model.predict(d)
    })
    
    return result
