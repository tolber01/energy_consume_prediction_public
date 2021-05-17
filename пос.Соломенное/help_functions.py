import pandas as pd

"""ПОКА НЕТ ДАННЫХ ЗА 2021 ГОД"""
def parse_weather(search_city,first_date,last_date):
    cities_df = pd.read_csv('cities.csv', delimiter=',')
    cities_df.loc[cities_df.city == search_city]
    url = str('http://pogoda-service.ru/archive_gsod_res.php?country=RS&station='+str(cities_df.loc[cities_df.city == search_city].station.item())+'&datepicker_beg='+first_date.strftime('%d.%m.%Y')+'&datepicker_end='+last_date.strftime('%d.%m.%Y'))
    df = pd.read_html(url)[0]
    df.columns = ['date','max_temp','min_temp','mid_temp','atmo','wind','rain','ef_temp']
    df['date'] = pd.to_datetime(df['date'])
    del df['atmo'], df['wind'],  df['rain'], df['ef_temp'] #атмосферное давление, скорость ветра, осадки, эффективная темп.; особо не нужны
    return df


