"""
Fallback Model Training Module for NASA Patent Forecasting (No TensorFlow)
This module provides a version that works without TensorFlow dependencies.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
import joblib
from typing import Dict, Tuple, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def create_ensemble_model_fallback(input_dim):
    """
    Create a StackingRegressor ensemble model using only scikit-learn models.
    
    The ensemble combines:
    - RandomForestRegressor: Ensemble of decision trees
    - XGBoost: Gradient boosting
    - SVR: Support vector regression
    
    Args:
        input_dim (int): Number of input features
        
    Returns:
        StackingRegressor: Configured ensemble model
    """
    
    print(f"Creating fallback ensemble model with input dimension: {input_dim}")
    
    # 1. Random Forest Regressor (replaces neural network)
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 2. XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    # 3. Support Vector Regressor
    svr_model = SVR(
        kernel='rbf',
        C=100,
        gamma='scale',
        epsilon=0.1
    )
    
    # Create the stacking ensemble
    ensemble = StackingRegressor(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('svr', svr_model)
        ],
        final_estimator=RandomForestRegressor(n_estimators=50, random_state=42),
        cv=5,
        n_jobs=-1
    )
    
    print("Fallback ensemble model created with: RandomForest, XGBoost, SVR")
    
    return ensemble


def train_single_topic_model_fallback(topic_data, topic_name, look_back_window=5, test_size=0.2):
    """
    Train a forecasting model for a single topic using fallback ensemble.
    
    Args:
        topic_data (tuple): (X, y) arrays for the topic
        topic_name (str): Name of the topic
        look_back_window (int): Size of the look-back window
        test_size (float): Proportion of data for testing
        
    Returns:
        tuple: (trained_model, scaler, metrics, predictions)
    """
    
    X, y = topic_data
    
    if len(X) == 0:
        print(f"Warning: No data available for {topic_name}")
        return None, None, None, None
    
    print(f"\nTraining fallback model for {topic_name}...")
    print(f"Dataset size: {len(X)} examples")
    
    # Step 1: Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 2: Split data into training and testing sets
    if len(X) < 10:
        X_train, X_test = X_scaled, X_scaled
        y_train, y_test = y, y
        print(f"Warning: Small dataset ({len(X)} examples). Using all data for training.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, shuffle=False
        )
        print(f"Training set: {len(X_train)} examples")
        print(f"Test set: {len(X_test)} examples")
    
    # Step 3: Create and train the ensemble model
    model = create_ensemble_model_fallback(input_dim=look_back_window)
    
    print("Training fallback ensemble model...")
    try:
        # Train the model
        model.fit(X_train, y_train)
        
        # Step 4: Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Step 5: Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"Model performance for {topic_name}:")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  R²: {r2:.3f}")
        
        # Create predictions dictionary
        predictions = {
            'y_true': y_test,
            'y_pred': y_pred,
            'test_indices': np.arange(len(X) - len(X_test), len(X))
        }
        
        return model, scaler, metrics, predictions
        
    except Exception as e:
        print(f"Error training model for {topic_name}: {e}")
        return None, None, None, None


def generate_forecast_fallback(model, scaler, last_sequence, forecast_years, last_year):
    """
    Generate forecasts using the trained fallback model.
    
    Args:
        model: Trained ensemble model
        scaler: Fitted scaler for features
        last_sequence (np.array): Last sequence of historical data
        forecast_years (int): Number of years to forecast
        last_year (int): Last year of historical data
        
    Returns:
        tuple: (forecast_years_list, forecast_values_list)
    """
    
    # Scale the last sequence
    last_sequence_scaled = scaler.transform(last_sequence.reshape(1, -1))
    
    forecasts = []
    current_sequence = last_sequence_scaled[0].copy()
    
    for i in range(forecast_years):
        # Predict next value
        next_value = model.predict(current_sequence.reshape(1, -1))[0]
        forecasts.append(max(0, next_value))  # Ensure non-negative forecasts
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1)
        # Scale the new value to add to sequence
        current_sequence[-1] = scaler.transform([[next_value]])[0][0] if hasattr(scaler, 'transform') else next_value
    
    # Generate forecast years
    forecast_years_list = list(range(last_year + 1, last_year + 1 + forecast_years))
    
    return forecast_years_list, forecasts


def prepare_sequences_for_training_fallback(trend_data, look_back_window):
    """
    Prepare time series data for training (fallback version).
    
    Args:
        trend_data (pd.Series): Time series data for a topic
        look_back_window (int): Number of previous time steps to use as features
        
    Returns:
        tuple: (X, y) arrays where X contains sequences and y contains targets
    """
    
    if len(trend_data) <= look_back_window:
        print(f"Warning: Not enough data points ({len(trend_data)}) for look_back_window ({look_back_window})")
        return np.array([]), np.array([])
    
    X, y = [], []
    
    for i in range(look_back_window, len(trend_data)):
        # Use previous 'look_back_window' values as features
        X.append(trend_data.iloc[i-look_back_window:i].values)
        # Current value is the target
        y.append(trend_data.iloc[i])
    
    return np.array(X), np.array(y)


def train_all_topics_fallback(trends_table, look_back_window=5, forecast_years=5):
    """
    Train forecasting models for all topics using fallback method.
    
    Args:
        trends_table (pd.DataFrame): DataFrame with years as index and topics as columns
        look_back_window (int): Number of previous years to use as features
        forecast_years (int): Number of years to forecast into the future
        
    Returns:
        tuple: (forecasts_dict, trained_models_dict)
    """
    
    print(f"\n{'='*60}")
    print("TRAINING FALLBACK FORECASTING MODELS")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Look-back window: {look_back_window} years")
    print(f"  Forecast horizon: {forecast_years} years")
    print(f"  Topics to train: {len(trends_table.columns)}")
    print(f"  Using fallback ensemble (no TensorFlow)")
    
    forecasts = {}
    trained_models = {}
    successful_trainings = 0
    
    last_year = trends_table.index.max()
    
    for topic_name in trends_table.columns:
        print(f"\n{'-'*40}")
        print(f"Processing {topic_name}")
        print(f"{'-'*40}")
        
        # Get time series data for this topic
        topic_series = trends_table[topic_name].dropna()
        
        if len(topic_series) < look_back_window + 2:
            print(f"Skipping {topic_name}: insufficient data ({len(topic_series)} points)")
            continue
        
        # Prepare training sequences
        X, y = prepare_sequences_for_training_fallback(topic_series, look_back_window)
        
        if len(X) == 0:
            print(f"Skipping {topic_name}: no training sequences generated")
            continue
        
        # Train model for this topic
        model, scaler, metrics, predictions = train_single_topic_model_fallback(
            (X, y), topic_name, look_back_window
        )
        
        if model is not None:
            # Generate forecasts
            last_sequence = topic_series.iloc[-look_back_window:].values
            forecast_years_list, forecast_values = generate_forecast_fallback(
                model, scaler, last_sequence, forecast_years, last_year
            )
            
            # Store results
            forecasts[topic_name] = {
                'forecast_years': forecast_years_list,
                'forecasts': forecast_values,
                'historical_data': topic_series.values.tolist(),
                'metrics': metrics,
                'last_historical_year': last_year
            }
            
            trained_models[topic_name] = {
                'model': model,
                'scaler': scaler
            }
            
            successful_trainings += 1
            
            # Display forecast summary
            print(f"Forecast for {topic_name}:")
            for year, value in zip(forecast_years_list[:3], forecast_values[:3]):
                print(f"  {year}: {value:.1f} patents")
            if len(forecast_years_list) > 3:
                print(f"  ... and {len(forecast_years_list)-3} more years")
        
        print(f"Status: {'SUCCESS' if model is not None else 'FAILED'}")
    
    print(f"\n{'='*60}")
    print("FALLBACK TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully trained: {successful_trainings} models")
    print(f"❌ Failed: {len(trends_table.columns) - successful_trainings} models")
    print(f"📈 Forecasts generated for {len(forecasts)} topics")
    
    if successful_trainings == 0:
        print("⚠️  WARNING: No models were successfully trained!")
    
    return forecasts, trained_models


if __name__ == "__main__":
    print("Fallback Model Trainer Module - Ready for use without TensorFlow!")