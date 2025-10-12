"""
Enhanced Model Trainer with Simple Forecasting
This version provides forecasts even with limited data using trend extrapolation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from typing import Dict, Tuple, Any

warnings.filterwarnings('ignore')


def simple_trend_forecast(time_series_data, forecast_years=5):
    """
    Create simple trend-based forecasts using linear extrapolation.
    
    Args:
        time_series_data (pd.Series): Historical data
        forecast_years (int): Number of years to forecast
        
    Returns:
        tuple: (forecast_years_list, forecast_values_list)
    """
    
    years = time_series_data.index.values
    values = time_series_data.values
    
    # Calculate trend using linear regression
    if len(values) >= 2:
        # Simple linear trend calculation
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)  # Linear fit
        slope, intercept = coeffs
        
        # Generate forecasts
        last_year = years[-1]
        forecast_years_list = list(range(last_year + 1, last_year + 1 + forecast_years))
        
        forecast_values = []
        for i, future_year in enumerate(forecast_years_list):
            # Extrapolate using trend
            future_x = len(values) + i
            forecast_value = slope * future_x + intercept
            
            # Ensure non-negative forecasts and apply some smoothing
            forecast_value = max(0, forecast_value)
            
            # Add some realistic bounds based on historical data
            historical_max = np.max(values)
            historical_avg = np.mean(values)
            
            # Cap extreme forecasts
            if forecast_value > historical_max * 2:
                forecast_value = historical_max * 1.5
            
            forecast_values.append(forecast_value)
    
    else:
        # If very limited data, use last value with slight variation
        last_value = values[-1] if len(values) > 0 else 1
        last_year = years[-1] if len(years) > 0 else 2027
        
        forecast_years_list = list(range(last_year + 1, last_year + 1 + forecast_years))
        forecast_values = [max(0, last_value + np.random.normal(0, 0.5)) for _ in forecast_years_list]
    
    return forecast_years_list, forecast_values


def create_simple_forecasts(trends_table, forecast_years=5):
    """
    Create forecasts for all topics using simple trend extrapolation.
    
    Args:
        trends_table (pd.DataFrame): Time-series data with topics as columns
        forecast_years (int): Number of years to forecast
        
    Returns:
        tuple: (forecasts_dict, success_count)
    """
    
    print(f"\n🔮 CREATING SIMPLE TREND FORECASTS")
    print(f"{'='*50}")
    print(f"Using trend extrapolation for {len(trends_table.columns)} topics")
    print(f"Forecast horizon: {forecast_years} years")
    
    forecasts = {}
    successful_forecasts = 0
    last_year = trends_table.index.max()
    
    for topic_name in trends_table.columns:
        try:
            print(f"\n📈 Processing {topic_name}...")
            
            # Get time series for this topic
            topic_series = trends_table[topic_name]
            
            # Create forecast using trend extrapolation
            forecast_years_list, forecast_values = simple_trend_forecast(
                topic_series, forecast_years
            )
            
            # Store forecast data
            forecasts[topic_name] = {
                'forecast_years': forecast_years_list,
                'forecasts': forecast_values,
                'historical_data': topic_series.values.tolist(),
                'last_historical_year': last_year,
                'method': 'trend_extrapolation',
                'metrics': {
                    'historical_avg': float(np.mean(topic_series.values)),
                    'historical_trend': 'growing' if topic_series.iloc[-1] > topic_series.iloc[0] else 'declining',
                    'forecast_avg': float(np.mean(forecast_values))
                }
            }
            
            successful_forecasts += 1
            
            # Display forecast summary
            print(f"  ✅ Created forecast: {forecast_values[0]:.1f} → {forecast_values[-1]:.1f} patents")
            print(f"  📊 Historical average: {np.mean(topic_series.values):.1f} patents")
            
        except Exception as e:
            print(f"  ❌ Failed to create forecast for {topic_name}: {e}")
            
            # Create a minimal fallback forecast
            forecasts[topic_name] = {
                'forecast_years': list(range(last_year + 1, last_year + 1 + forecast_years)),
                'forecasts': [1.0] * forecast_years,  # Default to 1 patent per year
                'historical_data': [0] * len(trends_table),
                'last_historical_year': last_year,
                'method': 'fallback',
                'metrics': {
                    'historical_avg': 0.0,
                    'historical_trend': 'stable',
                    'forecast_avg': 1.0
                }
            }
    
    print(f"\n{'='*50}")
    print(f"📊 FORECAST SUMMARY:")
    print(f"  ✅ Successful forecasts: {successful_forecasts}")
    print(f"  📈 Total topics with forecasts: {len(forecasts)}")
    print(f"  🎯 Success rate: {(successful_forecasts/len(trends_table.columns)*100):.1f}%")
    
    return forecasts, successful_forecasts


def enhanced_train_all_topics(trends_table, look_back_window=2, forecast_years=5):
    """
    Enhanced training function that always provides forecasts.
    
    Args:
        trends_table (pd.DataFrame): Time-series trends data
        look_back_window (int): Number of previous years to use (not used in simple version)
        forecast_years (int): Number of years to forecast
        
    Returns:
        tuple: (forecasts_dict, models_dict)
    """
    
    print(f"\n{'='*60}")
    print("🤖 ENHANCED FORECASTING SYSTEM")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  📊 Topics to process: {len(trends_table.columns)}")
    print(f"  📅 Historical data: {len(trends_table)} years ({trends_table.index.min()}-{trends_table.index.max()})")
    print(f"  🔮 Forecast horizon: {forecast_years} years")
    print(f"  🎯 Method: Simple trend extrapolation (robust for limited data)")
    
    # Create forecasts using simple trend extrapolation
    forecasts, successful_count = create_simple_forecasts(trends_table, forecast_years)
    
    # Create a dummy models dict (for compatibility)
    models = {topic: 'trend_extrapolation' for topic in forecasts.keys()}
    
    print(f"\n🎉 FORECASTING COMPLETED!")
    print(f"  📈 Generated forecasts for {len(forecasts)} topics")
    print(f"  ✅ Ready for patent evaluation!")
    
    return forecasts, models


if __name__ == "__main__":
    print("Enhanced Model Trainer - Ready for reliable forecasting!")