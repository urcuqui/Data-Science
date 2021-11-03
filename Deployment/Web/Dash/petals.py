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
    color="species")

app.layout = html.Div([
    html.Div(
        [
        html.H1(children="Datos"),
        dcc.Graph(id="fig1", figure=fig)
        ])

])



if __name__ == '__main__':
    app.run_server(debug=True)