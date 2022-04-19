from dash import html, Dash, dcc, dependencies, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from models import neural_network, gradient_boosting, random_forrest, train_test
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
prepared_data = pd.read_csv('../data/full_data_civil_building.csv', index_col=0)
app = Dash(__name__, external_stylesheets=external_stylesheets)

MODELS = ['gradient-boosting', 'neural-network', 'random-forrest']

app.layout = html.Div(children=[
    html.H1(children='IST Energy Monitor - Tobi`s Dashboard'),

    html.Div(children='''
        Display different features
    '''),

    html.Div([
        dcc.Dropdown(
            id='data-select',
            options=[{'label': i, 'value': i} for i in prepared_data.columns[1:]],  # not display index here
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

    html.Div([
        dcc.Dropdown(
            id='model-select',
            options=[{'label': i, 'value': i} for i in MODELS],
            value='random-forrest'
        ),
    ]),
    html.Button('Run Model', id='run-model', n_clicks=0),
    dcc.Loading(id="loading-icon",
                type='dot',
                children=[html.Div(dcc.Graph(id='model-graph'))]),

    html.Center([
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4("MSE", className="card-title1"),
                        html.P(
                            "Test",
                            className="mse-text",
                        )
                    ]
                ),
            ],
            style={"width": "18rem", "border": "2rem", 'display': 'inline-block'},
        ),
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4("MSE", className="card-title2"),
                        html.P(
                            "Test",
                            className="mse-text",
                        )
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
    Output('model-graph', 'figure'),
    Input('feature-select', 'value'),
    Input('model-select', 'value'),
    Input('run-model', 'n_clicks'))
def run_model(feature_select, model_type, n_clicks):
    if n_clicks > 0:
        print(n_clicks)
        print(model_type)
        print(feature_select)
        prediction_data = ''
        if model_type == 'random-forrest':
            x_train, y_train, x_test, y_test = train_test(prepared_data, features=feature_select)
            prediction_data = random_forrest(x_train, y_train, x_test, y_test)
        else:
            print("Error")

        return create_forecast_figure(prediction_data)
    else:
        raise PreventUpdate


def create_forecast_figure(prediction_data):
    return {
        'data': [
            {'x': prediction_data.index.to_list(), 'y': prediction_data[0], 'type': 'line', 'name': 'power'},
        ],
        'layout': {
            'title': 'IST hourly electricity consumption (kWh)'
        }
    }


def create_data_figure(column):
    return {
        'data': [
            {'x': prepared_data.index, 'y': prepared_data[column], 'type': 'line', 'name': column},
        ],
        'layout': {
            'title': 'Data'
        }
    }


if __name__ == '__main__':
    app.run_server(debug=True)
