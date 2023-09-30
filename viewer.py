import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dash_table import DataTable
from dash.dependencies import Input, Output

# Create a dummy DataFrame for the example
np.random.seed(42)
n_samples = 1000
n_assets = 7
returns_data = np.random.randn(n_samples, n_assets)
returns_df = pd.DataFrame(returns_data,
                          columns=['USDJPY', 'USDEUR', 'Oil', 'Gold', 'GermanBond', 'Nasdaq', 'USTreasuryBond'])
returns_df.index = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')

# Calculate rolling correlations across all asset combinations
correlation_matrix = returns_df.corr()

# Create a Dash web application
app = Dash(__name__)

# Define the layout of the web app
asset_pairs = [(asset1, asset2) for asset1 in returns_df.columns for asset2 in returns_df.columns if asset1 != asset2]

# Set the initial selected pairs as a list of tuples
initial_selected_pairs = [('USDJPY', 'USDEUR'), ('Oil', 'Gold')]

# Define the layout of the web app
app.layout = html.Div([
    html.H1("Interactive Financial Data Viewer"),

    # Pairwise copula visualizations
    dcc.Graph(id='copula-visualization'),

    # Rolling correlations heatmap (updated)
    html.H2("Correlation Matrix Heatmap"),  # Updated title
    dcc.Graph(id='correlation-heatmap'),  # Updated ID

    # Dropdown for choosing asset pairs
    html.H2("Choose Asset Pairs"),
    dcc.Dropdown(
        id='asset-pairs',
        options=[
            {'label': f'{pair[0]} vs. {pair[1]}', 'value': pair}
            for pair in asset_pairs
        ],
        multi=True,
        value=initial_selected_pairs  # Initial selected pairs
    ),

    # Last 10 days of data table
    html.H2("Last 10 Days of Data"),
    DataTable(
        id='last-10-days-data',
        columns=[{'name': col, 'id': col} for col in returns_df.columns],
        data=returns_df.tail(10).to_dict('records')
    )
])


# Define a function to calculate covariance ellipse
def calculate_covariance_ellipse(x, y, n_std=2):
    cov_matrix = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    major_axis_index = np.argmax(eigenvalues)
    major_axis = eigenvectors[:, major_axis_index]
    angle = np.arctan2(major_axis[1], major_axis[0])
    width = 2 * n_std * np.sqrt(eigenvalues[major_axis_index])
    height = 2 * n_std * np.sqrt(eigenvalues[1 - major_axis_index])
    return width, height, np.degrees(angle)


# Define callback functions for updating plots
@app.callback(
    Output('copula-visualization', 'figure'),
    Output('correlation-heatmap', 'figure'),  # Updated ID
    Input('last-10-days-data', 'selected_rows'),
    Input('asset-pairs', 'value')
)
def update_plots(selected_rows, selected_pairs):
    # Update the pairwise copula visualization based on selected rows
    selected_assets = list(returns_df.columns)
    copula_traces = []
    for asset1, asset2 in selected_pairs:
        x = returns_df[asset1].tail(10)
        y = returns_df[asset2].tail(10)
        copula_trace = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name=f'{asset1} vs. {asset2}',
        )
        # Calculate covariance ellipse
        width, height, angle = calculate_covariance_ellipse(x, y)
        ellipse_trace = go.Scatter(
            x=[np.mean(x)],
            y=[np.mean(y)],
            mode='lines',
            line=dict(
                color='black',
                width=2,
                shape='spline',  # Use 'spline' to approximate an ellipse
                smoothing=1.3,  # Adjust smoothing for a better fit
            ),
            showlegend=False,
        )
        copula_traces.extend([copula_trace, ellipse_trace])

    copula_fig = {
        'data': copula_traces,
        'layout': go.Layout(
            title='Pairwise Copula Visualization',
            xaxis={'title': 'Last 10 Days Returns'},
            yaxis={'title': 'Last 10 Days Returns'},
        )
    }

    # Update the correlation matrix heatmap (updated)
    correlation_heatmap_fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,  # Use the correlation matrix as the 'z' data
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis'
    ))

    correlation_heatmap_fig.update_layout(
        title='Correlation Matrix Heatmap',
        xaxis_title='Assets',
        yaxis_title='Assets'
    )

    return copula_fig, correlation_heatmap_fig  # Updated heatmap figure


# Run the web app
if __name__ == '__main__':
    app.run_server(debug=True)
