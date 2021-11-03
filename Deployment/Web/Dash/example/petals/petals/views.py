from django.http import HttpResponse
import dash
from django_plotly_dash import DjangoDash
import plotly.express as px
import dash_html_components as html
from django.shortcuts import render
import dash_core_components as dcc


app = DjangoDash("EDA")

df = px.data.iris()

fig = px.scatter_matrix(df,  dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
    color="species", symbol="species", labels={col:col.replace('_', ' ') for col in df.columns})

app.layout = html.Div([
    html.Div(
        [
        html.H1(children="Datos"),
        dcc.Graph(id="fig1", figure=fig),       
        ])
])



def index(request, *args, **kwargs):
  return render(request, 'index.html')

def eda(request):
  import seaborn as sns
  iris = sns.load_dataset('iris')
  print("WTF! kuky")
  return render(request, 'eda.html')
