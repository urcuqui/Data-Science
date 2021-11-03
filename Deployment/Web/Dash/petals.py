# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_html_components as html
import pandas as pd
import plotly.express as px
import dash_core_components as dcc
from dash.dependencies import Input, Output

import pandas as pd

app = dash.Dash(__name__)


df = px.data.iris()

fig = px.scatter_matrix(df,  dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
    color="species", symbol="species", labels={col:col.replace('_', ' ') for col in df.columns})

app.layout = html.Div([
    html.Div(
        [
        html.H1(children="Datos"),
        dcc.Graph(id="fig1", figure=fig),
        html.Hr()
        ], style={'height':'60%', 'width': '80%',' display': 'inline-block'}),
    html.Div([
        "Sepal width: ",
        dcc.Input(id='my-input-sepal-width', value='0', type='number'),
        "Sepal length: ",
        dcc.Input(id='my-input-sepal-length', value='0', type='number'),
        "Petal width: ",
        dcc.Input(id='my-input-petal-width', value='0', type='number'),
        "Petal length: ",
        dcc.Input(id='my-input-petal-length', value='0', type='number'),
        html.Br(),
        html.Br(),
        html.Div(id='my-output')
        ])

])

@app.callback(
    Output(component_id="my-output", component_property="children"),
    Input(component_id="my-input-sepal-width", component_property="value"),
    Input(component_id="my-input-sepal-length", component_property="value"),
    Input(component_id="my-input-petal-width", component_property="value"),
    Input(component_id="my-input-petal-length", component_property="value")
    )
def prediction_output_div(septal_width, sepal_length, petal_width, petal_length):
    import joblib

    loaded_model = joblib.load("model.joblib")

    result = loaded_model.predict([[septal_width,sepal_length,petal_width,petal_length]])

    return("Prediction: {}".format(result))

if __name__ == '__main__':
    app.run_server(debug=True)