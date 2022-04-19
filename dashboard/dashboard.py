from dash import html, Dash, dcc, Input, Output
import dash_bootstrap_components as dbc
from pickle import load
from models import neural_network, gradient_boosting, random_forrest, train_test
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import plotly.express as px

# data
with open('../data/dataframe_preprocessing.pickle', 'rb') as f:
    prepared_data = load(f)
start = datetime(year=2019, month=1, day=1)
end = datetime(year=2019, month=4, day=30)
prepared_data_2019 = prepared_data[start:end]
MODELS = ['gradient-boosting', 'neural-network', 'random-forrest']

# app
app = Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children="IST Energy Monitor 💻️ Tobi's Dashboard"),
    html.P('Hello Carlos! 🤓 Welcome to my Dashboard!'),
    html.P(' This dashboard can be used for a step by step analysis of the data '
           'and features to select. Special about this dashboard is that different features can be tried '
           'as the graphs and error metrics update live. This dashboard was styled with love and patience '
           'by hand using not a single external stylesheet which makes it look a bit boring but still feels '
           'intuitive and easy to use (function before form). Enjoy! '),
    html.Div([
        html.H2('Plot raw data'),
        html.H4('Pick column to display'),
        dcc.Dropdown(
            id='data-select',
            options=[{'label': i, 'value': i} for i in prepared_data.columns],
            value='Power_kW',
            style=dict(
                width='50%',
                display='inline-block',
                verticalAlign="middle")
        ),
    ]),

    html.H4('Change plot type'),
    dcc.RadioItems(
        id='plot-select',
        options=[
            {'label': 'Line Plot', 'value': 'line-plot'},
            {'label': 'Scatter Plot', 'value': 'scatter-plot'},
            {'label': 'Box Plot', 'value': 'box-plot'},
        ],
        value='line-plot'
    ),

    dcc.Graph(id='yearly-graph'),

    html.H2('Forecasting'),
    html.H3('Choose your features here'),
    html.Div([
        dcc.Dropdown(list(prepared_data.columns)[1:], ['holiday', 'temp_C', 'Power_kW_-1_day'], multi=True,
                     id='feature-select', style=dict(
                width='90%',
                display='inline-block',
                verticalAlign="middle")
                     )
    ]),

    html.H3('Choose your model here'),
    html.Div([
        dcc.Dropdown(
            id='model-select',
            options=[{'label': i, 'value': i} for i in MODELS],
            value='random-forrest',
            style=dict(
                width='50%',
                display='inline-block',
                verticalAlign="middle")
        ),
    ]),
    dcc.Loading(id="loading-icon",
                type='dot',
                children=[html.Div(dcc.Graph(id='model-graph'))]),

    html.H2('Error Metrics Evaluation'),
    html.Center([
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H3("MSE", className="card-title1"),
                        html.P(id='mse_output'),
                    ],

                ),
            ],
            style={"width": "18rem", "border": "2rem", 'display': 'inline-block'},
        ),
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H3("MAPE", className="card-title2"),
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
    Input('data-select', 'value'),
    Input('plot-select', 'value'), )
def update_graph(column, plot_tyoe):
    if plot_tyoe == 'line-plot':
        return px.line(prepared_data_2019, x=prepared_data_2019.index, y=column).update_layout(xaxis_title='time')
    if plot_tyoe == 'box-plot':
        return px.box(prepared_data_2019, x=column)
    if plot_tyoe == 'scatter-plot':
        return px.scatter(prepared_data_2019, x=prepared_data_2019.index, y=column)


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

    return create_forecast_figure(prediction_data, y_test), round(mse, 2), f'{str(round(mape * 100, 2))}%'


def create_forecast_figure(prediction_data, real_data):
    return {
        'data': [
            {'x': real_data.index.to_list(), 'y': real_data['Power_kW'], 'type': 'line', 'name': 'actual'},
            {'x': prediction_data.index.to_list(), 'y': prediction_data[0], 'type': 'line', 'name': 'prediction'}
        ],
        'layout': {
            'x_label': 'time',
            'y_label': 'power (kW)'
        }
    }


if __name__ == '__main__':
    app.run_server(debug=True)
