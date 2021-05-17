import statsmodels.api as sm
from matplotlib import pyplot as plt
import random
import pandas as pd
from pandas.tseries.offsets import DateOffset
from help_functions import parse_weather


"""
Функция SARIMAX без доп. переменных (метод для частных домов, когда нет информации о температуре)
PARAMETERS:
-----------------------------
y: {pd.DataFrame} 
    Исходные данные, где 2 колонки: 'date' и 'values'. Рекомендуем наблюдения за 3-4 месяца (сгрупированные по дням).
    В данных не должно быть пропусков.
param_seasonal: {int, int, int, int} 
    Параметры сезонности. Наилучший баланс между скоростью и качеством наблюдался при (1, 1, 1, S_WEEK), где S_WEEK - 7
OUTPUT:
------------------------------
pred: {pd.DataFrame}
    Предсказания с колонками 'date', 'forecast' (предсказание), 'lower values'(нижняя граница дов. интервала), 
    'upper values' (верхняя граница)
    Это датафрейм с предсказаниями до конца месяца. 
    Функция сама рассчитывает сколько дней осталось до конца месяца на основании последнего дня в выборке
sum_all: {int} сумма на конец месяца (с учетом уже известных наблюдений)
"""


def sarimax(y, param_seasonal):
    # распаковываем данные
    values = y["values"]
    date = y["date"]

    # находим день, с которого начать предсказание и на которм закончить
    start_forecasting = pd.to_datetime(date.iloc[-1].to_timestamp())+DateOffset(days=1)
    end_forecasting = start_forecasting.replace(day=1, month=start_forecasting.month+1)-DateOffset(days=1)
    pred = pd.DataFrame(pd.date_range(start_forecasting, end_forecasting).to_pydatetime(), columns=["date"])
    d = pred.shape[0]
    mod = sm.tsa.statespace.SARIMAX(values,
                                    seasonal_order=param_seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    full_output=False)
    # считаем сумму, которая уже есть
    sum_before = y[(pd.to_datetime(y["date"].apply(lambda x: x.to_timestamp())).apply(lambda x: x.month) == start_forecasting.month) & (pd.to_datetime(y["date"].apply(lambda x: x.to_timestamp())).apply(lambda x: x.year) == start_forecasting.year)]["values"].sum()
    results = mod.fit()
    pred_uc = results.get_forecast(steps=d)
    mean = pred_uc.predicted_mean.to_numpy()
    pred = pd.concat([pred, pred_uc.conf_int().reset_index(drop=True)], axis=1)
    pred["forecast"] = mean
    # потребление не может быть отрицательным
    pred.loc[pred['lower values'] < 0, 'lower values'] = 0
    pred.loc[pred['forecast'] < 0, 'forecast'] = 0
    # суммируем то, что было и что предсказали
    sum_all = sum_before + pred["forecast"].values.sum()
    return pred, sum_all


"""
Функция SARIMAX с доп. переменными (метод для частных домов, когда есть информация о температуре)
PARAMETERS:
---------------------
y: {pd.DataFrame}
    Исходные данные, где 2 колонки: 'date' и 'values'. Рекомендуем наблюдения за 3-4 месяца (сгрупированные по дням). 
    В данных не должно быть пропусков.
param_seasonal: {int, int, int, int} 
    Параметры сезонности. Наилучший баланс между скоростью и качеством наблюдался при (1, 1, 1, S_WEEK), где S_WEEK - 7
city: {str} 
    Город, где проводятся наблюдения (используется таблица с известными городами, если города нет в списке, то использовать обычную SARIMAX)
OUT:
----------------------
pred: {pd.DataFrame}
    Предсказание с колонками 'date', 'forecast' (предсказание), 'lower values'(нижняя граница дов. интервала), 'upper values' (верхняя граница)
    Это датафрейм с предсказаниями до конца месяца. Функция сама рассчитывает сколько дней осталось до конца месяца на основании последнего дня в выборке
sum_all: {int} сумма на конец месяца (с учетом уже известных наблюдений)

Функция сама парсит информацио о температуре, в случае ошибки с поиском температуры возвращает (None, -1)
"""


def sarimax_exog(y, param_seasonal, city):
    # распаковываем данные
    values = y["values"]
    date = y["date"]

    # находим день, с которого начать предсказание и на которм закончить
    start_forecasting = pd.to_datetime(date.iloc[-1].to_timestamp()) + DateOffset(days=1)
    end_forecasting = start_forecasting.replace(day=1, month=start_forecasting.month + 1) - DateOffset(days=1)
    pred = pd.DataFrame(pd.date_range(start_forecasting, end_forecasting).to_pydatetime(), columns=["date"])
    d = pred.shape[0] # кол-во дней для прогноза

    # Парсим температуру
    try:
        temp_df = parse_weather(city, date.iloc[0].to_timestamp(), end_forecasting)
    except:
        return None, -1

    # делим температуру на известную и предсказываемую
    temp_train, temp_test = temp_df["mid_temp"].iloc[:-d], temp_df["mid_temp"].iloc[-d:]

    mod = sm.tsa.statespace.SARIMAX(values,
                                    exog=temp_train.values,
                                    seasonal_order=param_seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    full_output=False)

    # считаем сумму, которая уже есть
    sum_before = y[(pd.to_datetime(y["date"].apply(lambda x: x.to_timestamp())).apply(
        lambda x: x.month) == start_forecasting.month) & (
                           pd.to_datetime(y["date"].apply(lambda x: x.to_timestamp())).apply(
                               lambda x: x.year) == start_forecasting.year)]["values"].sum()
    results = mod.fit()
    pred_uc = results.get_forecast(steps=d, exog=temp_test.values)
    mean = pred_uc.predicted_mean.to_numpy()
    pred = pd.concat([pred, pred_uc.conf_int().reset_index(drop=True)], axis=1)
    pred["forecast"] = mean
    # потребление не может быть отрицательным
    pred.loc[pred['lower values'] < 0, 'lower values'] = 0
    pred.loc[pred['forecast'] < 0, 'forecast'] = 0

    # суммируем то, что было и что предсказали
    sum_all = sum_before + pred["forecast"].values.sum()
    return pred, sum_all


if __name__ == "__main__":
    """ПРИМЕР РАБОТЫ (до черты-перевод данных в читабельный вид и пропуск данных, в которых есть пропуски)"""
    df = pd.read_csv("concat_data2.csv", na_values=['-'])
    data = df.iloc[random.choice(range(len(df))), ::]
    while data.isnull().values.any():
        data = df.iloc[random.choice(range(len(df))), ::]
    data = pd.DataFrame(data = {"date":df.columns, "values": data})[1:]
    "---------------------------------------------------------------"
    # приводим к нормальной форме даты и группируем по дням (этот кусок скорее всего понадобится)
    data['date'] = pd.to_datetime(data['date'])
    per = data['date'].dt.to_period("d")
    data = data.groupby(per)
    data_real = data.sum()[:-1]

    # параметры
    S_WEEK = 7
    IDX_START = -31 * 4  # 4 месяца с конца

    data_real.reset_index(level=0, inplace=True)
    # вычитаем 3 месяца, так как нет информации по температуре за 2021 год(вычитаем 20, чтобы было 20 прогнозных значений)
    data_t_cut = data_real[:-31*3-20]
    # выбираем отрезок, на котором будем обучаться
    y = data_t_cut[IDX_START::]
    # используем функцию
    pred, sum_all = sarimax(y, (1, 1, 1, S_WEEK))

    # отображение результатов
    ax = y.plot(x="date", y="values", label='observed', figsize=(14, 4))
    pred[["forecast", "date"]].plot(x="date", y="forecast", ax=ax, label='Forecast')

    ax.fill_between(pred["date"].values,
                     pred["lower values"].values,
                     pred["upper values"].values, color='k', alpha=.25)
    real_data = data_t_cut
    ax.set_xlabel('Дата')
    ax.set_ylabel('Потребление')
    plt.legend()
    plt.show()
    print(f"pred sarimax: {sum_all}")

    # то же самое
    pred, sum_all = sarimax_exog(y, (1, 1, 1, S_WEEK), 'Петрозаводск')
    ax = y.plot(x="date", y="values", label='observed', figsize=(14, 4))
    pred[["forecast", "date"]].plot(x="date", y="forecast", ax=ax, label='Forecast')

    ax.fill_between(pred["date"].values,
                    pred["lower values"].values,
                    pred["upper values"].values, color='k', alpha=.25)

    ax.set_xlabel('Дата')
    ax.set_ylabel('Потребление')
    plt.legend()
    plt.show()
    print(f"pred sarimax_exog: {sum_all}")

