from distutils.log import error
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt


def evaluate_forecasts(actual, predicted):
    errors = {}
    errors['mae'] = mean_absolute_error(y_true=actual, y_pred=predicted)
    errors['mse'] = mean_squared_error(y_true=actual, y_pred=predicted)
    errors['rmse'] = sqrt(mean_squared_error(y_true=actual, y_pred=predicted))
    errors['mape'] = mean_absolute_percentage_error(y_true=actual, y_pred=predicted)
    return errors
