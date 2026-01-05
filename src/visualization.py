"""
Interactive Plotly visualization utilities for FitPulse forecasting
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_interactive_forecast_chart(df, forecast, metric_name, unit=""):
    """
    Create an interactive Plotly forecast chart with actual values, predictions, and confidence intervals.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original data with columns ['ds', 'y']
    forecast : pd.DataFrame
        Prophet forecast output
    metric_name : str
        Name of the metric (e.g., "Heart Rate")
    unit : str
        Unit of measurement (e.g., "bpm", "steps")
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    
    # Create figure
    fig = go.Figure()
    
    # Add actual historical values
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Actual</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.1f} ' + unit + '<extra></extra>'
    ))
    
    # Add forecasted values
    forecast_future = forecast[forecast['ds'] > df['ds'].max()]
    
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#2ca02c', width=2, dash='dash'),
        hovertemplate='<b>Forecast</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.1f} ' + unit + '<extra></extra>'
    ))
    
    # Add confidence interval as shaded area
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='95% Confidence Interval',
        fillcolor='rgba(44, 160, 44, 0.2)',
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{metric_name} Forecast (Next 14 Days)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f77b4'}
        },
        xaxis_title='Date',
        yaxis_title=f'{metric_name} ({unit})',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(family='Arial, sans-serif', size=11),
        plot_bgcolor='rgba(240, 240, 245, 0.5)',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#e0e0e0',
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor='#e0e0e0',
            gridwidth=0.5,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='#e0e0e0',
            gridwidth=0.5,
            zeroline=False
        )
    )
    
    return fig


def create_forecast_summary_stats(df, forecast, metric_name, unit=""):
    """
    Calculate and return summary statistics for the forecast.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original data
    forecast : pd.DataFrame
        Prophet forecast output
    metric_name : str
        Name of the metric
    unit : str
        Unit of measurement
    
    Returns:
    --------
    dict
        Dictionary with summary statistics
    """
    
    actual_mean = df['y'].mean()
    actual_std = df['y'].std()
    
    forecast_future = forecast[forecast['ds'] > df['ds'].max()]
    forecast_mean = forecast_future['yhat'].mean()
    forecast_change = ((forecast_mean - actual_mean) / actual_mean * 100) if actual_mean != 0 else 0
    
    return {
        'actual_mean': actual_mean,
        'actual_std': actual_std,
        'forecast_mean': forecast_mean,
        'forecast_change': forecast_change,
        'metric_name': metric_name,
        'unit': unit
    }
