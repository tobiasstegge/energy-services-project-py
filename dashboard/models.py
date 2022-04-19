from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from src.ds_tools.utils import split_timeseries
from pandas import DataFrame


def train_test(df, features):
    train, test = split_timeseries(df, train_size=0.5)
    y_train = train[['Power_kW']]
    y_test = test[['Power_kW']]
    x_train = train[features]
    x_test = test[features]
    return x_train, y_train, x_test, y_test


def random_forrest(x_train, y_train, x_test, y_test):
    model = RandomForestRegressor(bootstrap=True, min_samples_leaf=1,
                                  n_estimators=20, min_samples_split=15, max_features='sqrt', max_depth=20)
    model.fit(x_train, y_train)
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
