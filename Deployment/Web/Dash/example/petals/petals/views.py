from django.http import HttpResponse
import dash
from django_plotly_dash import DjangoDash
import plotly.express as px
import dash_html_components as html

def index(request):
  return HttpResponse("Hello world!")

def eda(request):
  import seaborn as sns
  iris = sns.load_dataset('iris')
  print("WTF! kuky")
  return app
app = DjangoDash(__name__)

app.layout = html.Div([
    html.H1("KUKY")    

])