# Power Consumption Prediction via autoregression and linear models

Repository provides an instrument to forecasting power consumption in Russian business area. Our approach doesn't use specific variables allowed make forecasting with higher presicion. Also we don't use any heavy models like RNN because our main purpose was creating fast model work for linear time. 

As a result we developed 2 algorithms:

1) Algorithm based on linear Whinters model without a trend. Use it to forecast business consumption, where time series is stable. Best history (data needed to learning) - 3-4 mounts before.
2) SARIMAX algoritm. Use it for forecasting consumption of small houses where time series isn't stable. (Please don't use it to predict hourly consumption because algorithm could work so long). Best history (data needed to learning) - 3-4 mounts before. (You also can use weather parameters, please see file SARIMAX.py)

There are results we obtained:
Hourly consumption forecasting for a linear model:
MAE - 8%
![alt text](https://github.com/tolber01/energy_consume_prediction_public/blob/main/main/winters_best.jpg)
Monthly consumption forecating for a SARIMAX model with additional temperature information (using daily summation):
MAE - 20%
![alt text](https://github.com/tolber01/energy_consume_prediction_public/blob/main/main/sarimax_best.jpg)
