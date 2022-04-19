from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from pandas import DataFrame
from datetime import datetime


def train_test(df, features):
    start_train = datetime(year=2017, month=1, day=1)
    end_train = datetime(year=2018, month=12, day=31)
    start_predict = datetime(year=2019, month=1, day=1)
    end_predict = datetime(year=2019, month=3, day=31)
    train = df[start_train:end_train]
    predict = df[start_predict:end_predict]
    y_train = train[['Power_kW']]
    y_test = predict[['Power_kW']]
    x_train = train[features]
    x_test = predict[features]
    return x_train, y_train, x_test, y_test


def random_forrest(x_train, y_train, x_test, y_test):
    model = RandomForestRegressor(bootstrap=True, min_samples_leaf=1,
                                  n_estimators=20, min_samples_split=15, max_features='sqrt', max_depth=20)
    model.fit(x_train, y_train.values.ravel())
    prediction_random_forrest = DataFrame(model.predict(x_test))
    prediction_random_forrest.index = y_test.index
    return prediction_random_forrest


def gradient_boosting(x_train, y_train, x_test, y_test):
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train)
    prediction_gradient_boost = DataFrame(model.predict(x_test))
    prediction_gradient_boost.index = y_test.index
    return prediction_gradient_boost


def neural_network(x_train, y_train, x_test, y_test):
    model = MLPRegressor(hidden_layer_sizes=(10, 10, 10, 10))
    model.fit(x_train, y_train)
    prediction_mlp_regressor = DataFrame(model.predict(x_test))
    prediction_mlp_regressor.index = y_test.index
    return prediction_mlp_regressor
