import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List
import plotly.express as px
from datetime import datetime
from spy_option_core import AdvancedOptionsScreener


class OptionsVisualizer:
    """Class for creating interactive visualizations of options analysis results"""

    def __init__(self, analysis_results: Dict):
        """
        Initialize with analysis results from AdvancedOptionsScreener

        Parameters:
        -----------
        analysis_results: Dict
            Output from AdvancedOptionsScreener.analyze_options_chain()
        """
        self.results = analysis_results
        self.options_data = analysis_results['options_data']
        self.current_price = analysis_results['current_price']
        self.summary = analysis_results['summary']

    def plot_volatility_skew(self) -> go.Figure:
        """Create interactive volatility skew visualization"""
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Volatility Skew by Expiry',
                'Skew Term Structure',
                'Put-Call Skew Spread',
                'Butterfly Spread'
            ),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )

        # Plot 1: Volatility Skew
        for expiry in self.options_data['expiry'].unique()[:5]:  # Show first 5 expiries
            expiry_data = self.options_data[self.options_data['expiry'] == expiry]

            fig.add_trace(
                go.Scatter(
                    x=expiry_data['strike'],
                    y=expiry_data['impliedVolatility'],
                    name=f'Expiry {expiry}',
                    mode='markers'
                ),
                row=1, col=1
            )

        fig.add_vline(
            x=self.current_price,
            line_dash="dash",
            line_color="red",
            annotation_text="Current Price",
            row=1, col=1
        )

        # Plot 2: Skew Term Structure
        skew_data = self.summary.get('iv_skew', pd.DataFrame())
        if not skew_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=skew_data['days_to_expiry'],
                    y=skew_data['put_25d_vol'],
                    name='25Δ Put IV',
                    line=dict(color='red')
                ),
                row=1, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=skew_data['days_to_expiry'],
                    y=skew_data['call_25d_vol'],
                    name='25Δ Call IV',
                    line=dict(color='green')
                ),
                row=1, col=2
            )

            # Plot 3: Put-Call Skew Spread
            fig.add_trace(
                go.Scatter(
                    x=skew_data['days_to_expiry'],
                    y=skew_data['skew_spread'],
                    name='Put-Call Skew',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )

            # Plot 4: Butterfly Spread
            fig.add_trace(
                go.Scatter(
                    x=skew_data['days_to_expiry'],
                    y=skew_data['butterfly_spread'],
                    name='Butterfly Spread',
                    line=dict(color='purple')
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Volatility Skew Analysis",
            showlegend=True,
            template="plotly_white"
        )

        fig.update_xaxes(title_text="Strike Price", row=1, col=1)
        fig.update_xaxes(title_text="Days to Expiry", row=1, col=2)
        fig.update_xaxes(title_text="Days to Expiry", row=2, col=1)
        fig.update_xaxes(title_text="Days to Expiry", row=2, col=2)

        fig.update_yaxes(title_text="Implied Volatility", row=1, col=1)
        fig.update_yaxes(title_text="Implied Volatility", row=1, col=2)
        fig.update_yaxes(title_text="Skew Spread", row=2, col=1)
        fig.update_yaxes(title_text="Butterfly Spread", row=2, col=2)

        return fig

    def plot_volume_analysis(self) -> go.Figure:
        """Create volume analysis visualization"""
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Call Volume', 'Put Volume'))

        for opt_type, col in zip(['call', 'put'], [1, 2]):
            data = self.options_data[self.options_data['option_type'] == opt_type]

            fig.add_trace(
                go.Bar(
                    x=data['strike'],
                    y=data['volume'],
                    name=f'{opt_type.capitalize()} Volume'
                ),
                row=1, col=col
            )

        fig.update_layout(
            title='Options Volume Analysis',
            showlegend=True
        )

        return fig

    def plot_open_interest(self) -> go.Figure:
        """Create open interest visualization"""
        fig = make_subplots(rows=1, cols=1)

        for opt_type in ['call', 'put']:
            data = self.options_data[self.options_data['option_type'] == opt_type]

            fig.add_trace(
                go.Scatter(
                    x=data['strike'],
                    y=data['openInterest'],
                    name=f'{opt_type.capitalize()} OI',
                    mode='lines+markers'
                )
            )

        fig.update_layout(
            title='Open Interest Distribution',
            xaxis_title='Strike Price',
            yaxis_title='Open Interest',
            showlegend=True
        )

        return fig

    def plot_volatility_surface(self) -> go.Figure:
        """Create enhanced 3D volatility surface visualization with proper sizing and day 0"""
        # Add day 0 to the dataset
        current_data = self.options_data.copy()

        # Create day 0 data using current price
        day0_data = current_data[current_data['days_to_expiry'] == current_data['days_to_expiry'].min()].copy()
        day0_data['days_to_expiry'] = 0
        day0_data['impliedVolatility'] = day0_data['impliedVolatility'].mean()  # Use ATM volatility for day 0

        # Combine with original data
        plot_data = pd.concat([current_data, day0_data])

        # Create volatility surface with enhanced pivot
        surface = plot_data.pivot_table(
            values='impliedVolatility',
            index='strike',
            columns='days_to_expiry',
            aggfunc='mean'
        ).fillna(method='ffill')

        # Ensure proper ordering of strikes and expiries
        surface = surface.sort_index()
        surface = surface.reindex(sorted(surface.columns), axis=1)

        # Create the figure with larger size
        fig = go.Figure()

        # Add surface plot with enhanced styling
        fig.add_trace(go.Surface(
            x=surface.columns,  # Days to expiry
            y=surface.index,  # Strike prices
            z=surface.values,  # Implied volatilities
            colorscale='Viridis',
            name='Vol Surface',
            contours={
                "x": {"show": True, "start": 0, "end": surface.columns.max(), "size": 10, "color": "white"},
                "y": {"show": True, "start": surface.index.min(), "end": surface.index.max(), "size": 5,
                      "color": "white"},
                "z": {"show": True, "start": surface.values.min(), "end": surface.values.max(), "size": 0.02,
                      "color": "white"}
            }
        ))

        # Add current price marker with enhanced visibility
        fig.add_trace(go.Scatter3d(
            x=[0],  # At day 0
            y=[self.current_price],
            z=[surface.values.mean()],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond'
            ),
            name='Current Price'
        ))

        # Enhanced layout with better sizing and viewpoint
        fig.update_layout(
            title={
                'text': 'Implied Volatility Surface',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            scene={
                'xaxis_title': 'Days to Expiry',
                'yaxis_title': 'Strike Price',
                'zaxis_title': 'Implied Volatility',
                'camera': {
                    'up': {'x': 0, 'y': 0, 'z': 1},
                    'center': {'x': 0, 'y': 0, 'z': -0.1},
                    'eye': {'x': 1.5, 'y': 1.5, 'z': 1.2}
                },
                'aspectratio': {'x': 1, 'y': 1, 'z': 0.7},
                'xaxis': {'range': [-1, surface.columns.max()]},  # Start from -1 to show day 0 clearly
                'yaxis': {'range': [surface.index.min(), surface.index.max()]},
                'zaxis': {'range': [surface.values.min(), surface.values.max()]}
            },
            width=1000,  # Increased width
            height=800,  # Increased height
            margin=dict(l=65, r=50, b=65, t=90),
            template="plotly_white",
            showlegend=True
        )

        return fig

    def plot_volatility_surface_only(self) -> go.Figure:
        """Create 3D volatility surface visualization"""
        fig = go.Figure()

        # Create volatility surface
        surface = self.options_data.pivot_table(
            values='impliedVolatility',
            index='strike',
            columns='days_to_expiry',
            aggfunc='mean'
        ).fillna(method='ffill')

        fig.add_trace(go.Surface(
            x=surface.columns,  # Days to expiry
            y=surface.index,  # Strike prices
            z=surface.values,  # Implied volatilities
            colorscale='Viridis',
            name='Vol Surface'
        ))

        # Add current price marker
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[self.current_price],
            z=[self.options_data['impliedVolatility'].mean()],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Current Price'
        ))

        fig.update_layout(
            title='Implied Volatility Surface',
            scene=dict(
                xaxis_title='Days to Expiry',
                yaxis_title='Strike Price',
                zaxis_title='Implied Volatility',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            template="plotly_white"
        )

        return fig

    def plot_pca_analysis(self) -> go.Figure:
        """Visualize PCA components of volatility surface"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'First Principal Component',
                'Second Principal Component',
                'Explained Variance Ratio',
                'PC Loading Structure'
            )
        )

        # Extract PCA results
        vol_surface_pc1 = self.summary.get('vol_surface_pc1')
        vol_surface_pc2 = self.summary.get('vol_surface_pc2')

        if vol_surface_pc1 is not None and vol_surface_pc2 is not None:
            # Plot explained variance
            fig.add_trace(
                go.Bar(
                    x=['PC1', 'PC2', 'Other'],
                    y=[vol_surface_pc1, vol_surface_pc2, 1 - vol_surface_pc1 - vol_surface_pc2],
                    name='Explained Variance'
                ),
                row=2, col=1
            )

            # Add PC structure visualization
            strikes = sorted(self.options_data['strike'].unique())
            pc1_structure = np.exp(-0.5 * ((strikes - self.current_price) / self.current_price) ** 2)
            pc2_structure = ((strikes - self.current_price) / self.current_price) * pc1_structure

            fig.add_trace(
                go.Scatter(x=strikes, y=pc1_structure, name='PC1 Structure'),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=strikes, y=pc2_structure, name='PC2 Structure'),
                row=1, col=2
            )

            # Add PC loading heatmap
            pc_loadings = np.outer(pc1_structure, pc2_structure)
            fig.add_trace(
                go.Heatmap(
                    z=pc_loadings,
                    x=strikes,
                    y=strikes,
                    colorscale='RdBu',
                    name='PC Loadings'
                ),
                row=2, col=2
            )

        fig.update_layout(
            height=800,
            title_text="PCA Analysis of Volatility Surface",
            showlegend=True,
            template="plotly_white"
        )

        return fig

    def plot_term_structure(self) -> go.Figure:
        """Visualize volatility term structure"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Term Structure',
                'Term Structure Slope & Curvature'
            )
        )

        # Plot term structure
        term_structure = self.options_data.groupby('days_to_expiry')['impliedVolatility'].mean()

        fig.add_trace(
            go.Scatter(
                x=term_structure.index,
                y=term_structure.values,
                mode='lines+markers',
                name='Term Structure'
            ),
            row=1, col=1
        )

        # Plot fitted curve if available
        slope = self.summary.get('term_structure_slope')
        curvature = self.summary.get('term_structure_curvature')

        if slope is not None and curvature is not None:
            x = np.linspace(term_structure.index.min(), term_structure.index.max(), 100)
            y = curvature * x ** 2 + slope * x + term_structure.mean()

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name='Fitted Curve',
                    line=dict(dash='dash')
                ),
                row=1, col=1
            )

            # Plot slope and curvature
            fig.add_trace(
                go.Bar(
                    x=['Slope', 'Curvature'],
                    y=[slope, curvature],
                    name='Term Structure Parameters'
                ),
                row=1, col=2
            )

        fig.update_layout(
            height=500,
            title_text="Volatility Term Structure Analysis",
            showlegend=True,
            template="plotly_white"
        )

        return fig

    def plot_trading_opportunities(self) -> go.Figure:
        """Visualize top trading opportunities"""
        top_ops = self.summary.get('top_opportunities', pd.DataFrame())

        if top_ops.empty:
            return go.Figure()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Value Score by Strike',
                'Risk-Reward Analysis',
                'Opportunity Distribution',
                'Strategy Analysis'
            )
        )

        # Plot 1: Value Score by Strike
        fig.add_trace(
            go.Scatter(
                x=top_ops['strike'],
                y=top_ops['value_score'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=top_ops['impliedVolatility'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=top_ops.apply(
                    lambda x: f"Strike: {x['strike']}<br>Type: {x['option_type']}<br>Score: {x['value_score']:.2f}",
                    axis=1
                ),
                name='Value Score'
            ),
            row=1, col=1
        )

        # Plot 2: Risk-Reward
        fig.add_trace(
            go.Scatter(
                x=top_ops['theta'].abs(),
                y=top_ops['probability_itm'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=top_ops['value_score'],
                    colorscale='Viridis'
                ),
                text=top_ops.apply(
                    lambda x: f"Strike: {x['strike']}<br>Type: {x['option_type']}<br>Score: {x['value_score']:.2f}",
                    axis=1
                ),
                name='Risk-Reward'
            ),
            row=1, col=2
        )

        # Plot 3: Opportunity Distribution
        fig.add_trace(
            go.Histogram(
                x=top_ops['value_score'],
                nbinsx=20,
                name='Score Distribution'
            ),
            row=2, col=1
        )

        # Plot 4: Strategy Analysis
        strategy_scores = pd.DataFrame({
            'Strategy': ['Directional', 'Volatility', 'Income'],
            'Score': [
                top_ops['delta'].abs().mean(),
                top_ops['vega'].abs().mean(),
                top_ops['theta'].abs().mean()
            ]
        })

        fig.add_trace(
            go.Bar(
                x=strategy_scores['Strategy'],
                y=strategy_scores['Score'],
                name='Strategy Scores'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text="Trading Opportunities Analysis",
            showlegend=True,
            template="plotly_white"
        )

        return fig

    def save_all_visualizations(self, output_dir: str = 'option_analysis_plots'):
        """Save all visualizations to HTML files"""
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create and save all plots
        plots = {
            'volatility_skew': self.plot_volatility_skew(),
            'volatility_surface': self.plot_volatility_surface(),
            'pca_analysis': self.plot_pca_analysis(),
            'term_structure': self.plot_term_structure(),
            'trading_opportunities': self.plot_trading_opportunities()
        }

        for name, fig in plots.items():
            filename = f"{name}_{timestamp}.html"
            fig.write_html(os.path.join(output_dir, filename))

        return f"Plots saved to {output_dir}"


def display_vol_surface(st, fig):
    """Helper function to display volatility surface with proper sizing in Streamlit"""
    # Create a container with custom CSS for the plot
    plot_container = st.container()
    with plot_container:
        st.markdown("""
            <style>
                .volsurface-container {
                    width: 100%;
                    height: 800px;
                }
            </style>
        """, unsafe_allow_html=True)

        # Display the plot in the container
        st.markdown('<div class="volsurface-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, height=800)
        st.markdown('</div>', unsafe_allow_html=True)
def create_app():
    st.set_page_config(page_title="Options Analysis Dashboard", layout="wide")

    st.title("Options Analysis Dashboard")

    # Sidebar inputs
    st.sidebar.header("Settings")
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="SPY").upper()

    # Analysis selection
    st.sidebar.header("Select Analyses")
    analyses = st.sidebar.multiselect(
        "Choose analyses to display",
        ["Volatility Skew",
         "Volume Analysis",
         "Open Interest",
         "Trading Opportunity",
         "PCA Analysis",
         "Volatility Surface",
         "Term Structure"],
        default=["Volatility Skew"]
    )

    # Layout selection
    layout_type = st.sidebar.radio(
        "Select Layout",
        ["Vertical", "Grid"]
    )

    if st.sidebar.button("Analyze"):
        try:
            # Fetch and analyze data
            analyzer = AdvancedOptionsScreener(ticker)
            results = analyzer.analyze_options_chain()

            if results['options_data'].empty:
                st.error(f"No options data available for {ticker}")
                return

            visualizer = OptionsVisualizer(results)

            # Create plots based on selected analyses
            plots = []
            for analysis in analyses:#TODO create a map
                if "Volatility Skew" == analysis:
                    plt = visualizer.plot_volatility_skew()
                if "Volume Analysis" == analysis:
                    plt = visualizer.plot_volume_analysis()
                if "Open Interest" == analysis:
                    plt = visualizer.plot_open_interest()
                if "Trading Opportunity" == analysis:
                    plt = visualizer.plot_trading_opportunities()
                if "PCA Analysis" == analysis:
                    plt = visualizer.plot_pca_analysis()
                if "Volatility Surface" == analysis:
                    plt = visualizer.plot_volatility_surface()
                if "Term Structure" == analysis:
                    plt = visualizer.plot_term_structure()

                plots.append((analysis, plt))
            # Display plots based on layout
            if layout_type == "Vertical":
                for title, fig in plots:
                    st.subheader(title)
                    if "Volatility Surface" == title:
                        display_vol_surface(st, fig)
                    else:
                        st.plotly_chart(fig, use_container_width=True)
            else:  # Grid layout
                cols = st.columns(min(len(plots), 2))
                for idx, (title, fig) in enumerate(plots):
                    with cols[idx % 2]:
                        st.subheader(title)
                        if "Volatility Surface" == title:
                            display_vol_surface(st, fig)
                        else:
                            st.plotly_chart(fig, use_container_width=True)

            # Display summary statistics
            st.sidebar.header("Summary Statistics")
            st.sidebar.write(f"Current Price: ${results['current_price']:.2f}")

        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")


def plot_test():
    """Example usage of the visualization module"""
    from spy_option_core import AdvancedOptionsScreener

    # Get options analysis
    screener = AdvancedOptionsScreener()
    analysis = screener.analyze_options_chain()

    # Create visualizations
    viz = OptionsVisualizer(analysis)

    # Save all plots
    viz.save_all_visualizations()

    # merge all plots in one
    all_figs = []
    # Or display individual plots
    trade_fig = viz.plot_trading_opportunities()
    all_figs.append(trade_fig)
    trade_fig.show()

    surface_fig = viz.plot_volatility_surface()
    all_figs.append(surface_fig)
    surface_fig.show()

    skew_fig = viz.plot_volatility_skew()
    all_figs.append(skew_fig)
    skew_fig.show()

    pca_fig = viz.plot_pca_analysis()
    all_figs.append(pca_fig)
    pca_fig.show()

if __name__ == "__main__":
    #plot_test()
    create_app()