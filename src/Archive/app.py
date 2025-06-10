# app with no overlay
import dash
from dash import dcc, html
import plotly.graph_objs as go
import numpy as np
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Generate random probability distributions
np.random.seed(42)
x1 = np.random.normal(0, 1, 1000)
x2 = np.random.normal(2, 1.5, 1000)
x3 = np.random.exponential(1, 1000)
x4 = np.random.uniform(-2, 2, 1000)
x_sum = x1 + x2 + x3 + x4

df = pd.DataFrame({'Variable 1': x1, 'Variable 2': x2, 'Variable 3': x3, 'Variable 4': x4, 'Sum': x_sum})

colors = ['red', 'green', 'purple', 'orange']

def create_histogram(data, title, color='blue', height=250, customdata=None):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data, nbinsx=50, marker=dict(color=color),
        hoverinfo='x',
        hovertemplate=(
            "Sum: %{x}<br>" +
            "Variable 1: %{customdata[0]}<br>" +
            "Variable 2: %{customdata[1]}<br>" +
            "Variable 3: %{customdata[2]}<br>" +
            "Variable 4: %{customdata[3]}<extra></extra>"
        ) if customdata is not None else None,
        customdata=customdata
    ))
    fig.update_layout(title=title, margin=dict(l=40, r=40, t=40, b=40), height=height)
    return dcc.Graph(figure=fig, style={'height': f'{height}px'})

# Layout
app.layout = html.Div([
    html.Div([
        create_histogram(x1, "Variable 1", color=colors[0]),
        create_histogram(x2, "Variable 2", color=colors[1]),
        create_histogram(x3, "Variable 3", color=colors[2]),
        create_histogram(x4, "Variable 4", color=colors[3]),
    ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top', 'height': '100vh', 'overflow': 'auto'}),

    html.Div([
        create_histogram(df['Sum'], "Sum of Variables", height=1000, customdata=df[['Variable 1', 'Variable 2', 'Variable 3', 'Variable 4']].values),
    ], style={'width': '66%', 'display': 'inline-block', 'vertical-align': 'top', 'height': '100vh', 'overflow': 'auto'})
], style={'height': '100vh', 'overflow': 'hidden', 'display': 'flex'})

if __name__ == '__main__':
    app.run_server(debug=True)
