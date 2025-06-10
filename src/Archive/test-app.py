# App with overlay
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

df = pd.DataFrame({'GDP': x1, 'Ginni Index': x2, 'Producer Price Index': x3, 'Unemployment Rate': x4, 'Game Outcomes': x_sum})

colors = ['red', 'green', 'purple', 'orange']

def create_histogram(data, title, color='blue', height=250, customdata=None, opacity=1.0):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data, nbinsx=50, marker=dict(color=color, opacity=opacity),
        hoverinfo='x',
        hovertemplate=(
            "Game Outcomes: %{x}<br>" +
            "GDP: %{customdata[0]}<br>" +
            "Ginni Index: %{customdata[1]}<br>" +
            "Producer Price Index: %{customdata[2]}<br>" +
            "Unemployment Rate: %{customdata[3]}<extra></extra>"
        ) if customdata is not None else None,
        customdata=customdata
    ))
    fig.update_layout(title=title, margin=dict(l=40, r=40, t=40, b=40), height=height)
    return dcc.Graph(figure=fig, style={'height': f'{height}px'})

# Create the sum histogram with background histograms
def create_sum_histogram(df, height=1000):
    fig = go.Figure()
    
    # Add background histograms for each variable
    for i, var in enumerate(['GDP', 'Ginni Index', 'Producer Price Index', 'Unemployment Rate']):
        fig.add_trace(go.Histogram(
            x=df[var], nbinsx=50, marker=dict(color=colors[i], opacity=0.3), name=var, showlegend=True
        ))
    
    # Add the sum histogram on top
    fig.add_trace(go.Histogram(
        x=df['Game Outcomes'], nbinsx=50, marker=dict(color='blue', opacity=1.0), name='Sum',
        hovertemplate=(
            "Game Outcomes: %{x}<br>" +
            "GDP: %{customdata[0]}<br>" +
            "Ginni Index: %{customdata[1]}<br>" +
            "Producer Price Index: %{customdata[2]}<br>" +
            "Unemployment Rate: %{customdata[3]}<extra></extra>"
        ),
        customdata=df[['GDP', 'Ginni Index', 'Producer Price Index', 'Unemployment Rate']].values
    ))
    
    fig.update_layout(title="Game Outcomes", margin=dict(l=40, r=40, t=40, b=40), height=height, barmode='overlay')
    return dcc.Graph(figure=fig, style={'height': f'{height}px'})

# Layout
app.layout = html.Div([
    html.Div([
        create_histogram(x1, "GDP", color=colors[0]),
        create_histogram(x2, "Ginni Index", color=colors[1]),
        create_histogram(x3, "Producer Price Index", color=colors[2]),
        create_histogram(x4, "Unemployment Rate", color=colors[3]),
    ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top', 'height': '100vh', 'overflow': 'auto'}),

    html.Div([
        create_sum_histogram(df, height=1000),
    ], style={'width': '66%', 'display': 'inline-block', 'vertical-align': 'top', 'height': '100vh', 'overflow': 'auto'})
], style={'height': '100vh', 'overflow': 'hidden', 'display': 'flex'})

if __name__ == '__main__':
    app.run_server(debug=True)