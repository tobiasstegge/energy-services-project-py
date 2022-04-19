from dash import html, Dash, dcc, dependencies, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from pickle import load
from models import neural_network, gradient_boosting, random_forrest, train_test
from datetime import datetime
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

with open('../data/dataframe_preprocessing.pickle', 'rb') as f:
    prepared_data = load(f)

MODELS = ['gradient-boosting', 'neural-network', 'random-forrest']
mae = 0
mse = 0
rmse = 0
mape = 0

app.layout = html.Div(children=[
    html.H1(children='IST Energy Monitor - Tobi`s Dashboard'),

    html.Div(children='''
        Display different features
    '''),

    html.Div([
        dcc.Dropdown(
            id='data-select',
            options=[{'label': i, 'value': i} for i in prepared_data.columns],  # not display index here
            value='Power_kW'
        ),
    ]),

    dcc.Graph(id='yearly-graph'),

    # MODEL #
    html.H3('Choose your features here'),
    html.Div([
        dcc.Dropdown(list(prepared_data.columns)[1:], ['holiday', 'temp_C', 'Power_kW_-1_day'], multi=True,
                     id='feature-select')
    ]),

    html.H3('Choose your model here'),
    html.Div([
        dcc.Dropdown(
            id='model-select',
            options=[{'label': i, 'value': i} for i in MODELS],
            value='random-forrest'
        ),
    ]),
    dcc.Loading(id="loading-icon",
                type='dot',
                children=[html.Div(dcc.Graph(id='model-graph'))]),

    html.Center([
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4("MSE", className="card-title1"),
                        html.P(id='mse_output')
                    ]
                ),
            ],
            style={"width": "18rem", "border": "2rem", 'display': 'inline-block'},
        ),
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4("MAPE", className="card-title2"),
                        html.P(id='mape_output')
                    ]
                ),
            ],
            style={"width": "18rem", "border": "2rem", 'display': 'inline-block'},
        )]
    )

])


@app.callback(
    Output('yearly-graph', 'figure'),
    Input('data-select', 'value'))
def update_graph(column):
    return create_data_figure(column)


@app.callback(
    [Output('model-graph', 'figure'),
     Output('mse_output', 'children'),
     Output('mape_output', 'children')],
    [Input('feature-select', 'value'),
     Input('model-select', 'value')])
def run_model(feature_select, model_type):
    prediction_data = ''
    x_train, y_train, x_test, y_test = train_test(prepared_data, features=feature_select)
    if model_type == 'random-forrest':
        prediction_data = random_forrest(x_train, y_train, x_test, y_test)
    if model_type == 'gradient-boosting':
        prediction_data = gradient_boosting(x_train, y_train, x_test, y_test)
    if model_type == 'neural-network':
        prediction_data = neural_network(x_train, y_train, x_test, y_test)

    mse = mean_squared_error(y_true=y_test, y_pred=prediction_data)
    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=prediction_data)

    return create_forecast_figure(prediction_data, y_test), round(mse, 2), f'{str(round(mape * 100,2))}%'


def create_forecast_figure(prediction_data, real_data):
    return {
        'data': [
            {'x': real_data.index.to_list(), 'y': real_data['Power_kW'], 'type': 'line', 'name': 'actual'},
            {'x': prediction_data.index.to_list(), 'y': prediction_data[0], 'type': 'line', 'name': 'prediction'}
        ],
        'layout': {
            'title': 'IST hourly electricity consumption (kWh)',
            'x_label': 'time',
            'y_label': 'power (kW)'
        }
    }


def create_data_figure(column):
    start = datetime(year=2019, month=1, day=1)
    end = datetime(year=2019, month=4, day=30)
    prepared_data_2019 = prepared_data[start:end]
    return {
        'data': [
            {'x': prepared_data_2019.index, 'y': prepared_data_2019[column], 'type': 'line', 'name': column},
        ],
        'layout': {
            'title': 'Raw Data of 2019 (January - April)',
            'xaxis_title': 'time',
            'yaxis_title': column
        }
    }


if __name__ == '__main__':
    app.run_server(debug=True)
