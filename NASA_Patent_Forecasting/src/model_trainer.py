"""
Model Training Module for NASA Patent Forecasting
This module implements advanced ensemble models using StackingRegressor with ANN, XGBoost, and SVR.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasRegressor
import warnings
import joblib
from typing import Dict, Tuple, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


def create_ann_model(input_dim):
    """
    Create an Artificial Neural Network (ANN) model for regression.
    
    This function builds a deep neural network with multiple hidden layers,
    dropout for regularization, and appropriate activation functions.
    
    Args:
        input_dim (int): Number of input features (look_back window size)
        
    Returns:
        keras.Model: Compiled neural network model
    """
    
    # Build the neural network architecture
    model = keras.Sequential([
        # Input layer with normalization
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # First hidden layer
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Second hidden layer
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),
        
        # Output layer for regression
        layers.Dense(1, activation='linear')
    ])
    
    # Compile the model with appropriate optimizer and loss function
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model


def create_ensemble_model(input_dim):
    """
    Create a StackingRegressor ensemble model with ANN, XGBoost, and SVR as base models.
    
    The ensemble combines the strengths of different algorithms:
    - ANN: Captures complex non-linear patterns
    - XGBoost: Excellent gradient boosting performance
    - SVR: Robust support vector regression
    
    Args:
        input_dim (int): Number of input features
        
    Returns:
        StackingRegressor: Configured ensemble model
    """
    
    print(f"Creating ensemble model with input dimension: {input_dim}")
    
    # Create base models
    
    # 1. Artificial Neural Network (wrapped for sklearn compatibility)
    ann_model = KerasRegressor(
        model=create_ann_model,
        model__input_dim=input_dim,
        epochs=100,
        batch_size=32,
        verbose=0,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
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
    # The meta-learner (final estimator) is a simple linear regressor
    ensemble = StackingRegressor(
        estimators=[
            ('ann', ann_model),
            ('xgb', xgb_model),
            ('svr', svr_model)
        ],
        final_estimator=SVR(kernel='linear', C=1.0),
        cv=5,  # 5-fold cross-validation for stacking
        n_jobs=-1  # Use all available processors
    )
    
    print("Ensemble model created with base models: ANN, XGBoost, SVR")
    
    return ensemble


def train_single_topic_model(topic_data, topic_name, look_back_window=5, test_size=0.2):
    """
    Train a forecasting model for a single topic.
    
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
    
    print(f"\nTraining model for {topic_name}...")
    print(f"Dataset size: {len(X)} examples")
    
    # Step 1: Scale the features
    # MinMaxScaler normalizes features to [0,1] range for better model performance
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Also scale targets for neural network training
    y_reshaped = y.reshape(-1, 1)
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y_reshaped).flatten()
    
    # Step 2: Split data into training and testing sets
    if len(X) < 10:  # Use all data if dataset is very small
        X_train, X_test = X_scaled, X_scaled
        y_train, y_test = y_scaled, y_scaled
        print(f"Warning: Small dataset ({len(X)} examples). Using all data for training.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=42, shuffle=False
        )
        print(f"Training set: {len(X_train)} examples")
        print(f"Test set: {len(X_test)} examples")
    
    # Step 3: Create and train the ensemble model
    model = create_ensemble_model(input_dim=look_back_window)
    
    print("Training ensemble model...")
    try:
        # Train the model
        model.fit(X_train, y_train)
        
        # Step 4: Make predictions on test set
        y_pred_scaled = model.predict(X_test)
        
        # Transform predictions back to original scale
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Step 5: Calculate evaluation metrics
        mse = mean_squared_error(y_test_original, y_pred)
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        print(f"Model performance for {topic_name}:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.3f}")
        
        # Prepare predictions data
        predictions = {
            'y_true': y_test_original,
            'y_pred': y_pred
        }
        
        return model, (scaler, y_scaler), metrics, predictions
        
    except Exception as e:
        print(f"Error training model for {topic_name}: {e}")
        return None, None, None, None


def iterative_forecast(model, scaler_tuple, last_sequence, forecast_years=5):
    """
    Perform iterative forecasting for multiple years into the future.
    
    This method uses the model's own predictions as input for subsequent predictions,
    allowing us to forecast multiple years ahead.
    
    Args:
        model: Trained forecasting model
        scaler_tuple: (X_scaler, y_scaler) for data transformation
        last_sequence (np.array): Last sequence of historical data
        forecast_years (int): Number of years to forecast
        
    Returns:
        np.array: Array of forecasted values
    """
    
    scaler, y_scaler = scaler_tuple
    
    # Initialize with the last known sequence
    current_sequence = last_sequence.copy()
    forecasts = []
    
    print(f"Starting iterative forecast for {forecast_years} years...")
    
    for year in range(forecast_years):
        # Scale the current sequence
        current_sequence_scaled = scaler.transform(current_sequence.reshape(1, -1))
        
        # Make prediction
        prediction_scaled = model.predict(current_sequence_scaled)
        
        # Transform back to original scale
        prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
        
        # Ensure non-negative prediction (patent counts can't be negative)
        prediction = max(0, prediction)
        
        forecasts.append(prediction)
        
        # Update sequence for next prediction (sliding window)
        # Remove first element and add the new prediction
        current_sequence = np.append(current_sequence[1:], prediction)
    
    return np.array(forecasts)


def train_all_topics(trends_data, look_back_window=5, forecast_years=5):
    """
    Train forecasting models for all topics and generate forecasts.
    
    This is the main function that orchestrates the training process for all topics:
    1. Prepares supervised datasets for each topic
    2. Trains ensemble models
    3. Performs forecasting
    4. Compiles results
    
    Args:
        trends_data (pd.DataFrame): Time-series DataFrame with topics as columns
        look_back_window (int): Number of historical years to use as features
        forecast_years (int): Number of years to forecast into the future
        
    Returns:
        tuple: (forecasts_dict, trained_models_dict)
    """
    
    print("="*80)
    print("TRAINING FORECASTING MODELS FOR ALL TOPICS")
    print("="*80)
    
    print(f"Configuration:")
    print(f"  Look-back window: {look_back_window} years")
    print(f"  Forecast horizon: {forecast_years} years")
    print(f"  Number of topics: {len(trends_data.columns)}")
    print(f"  Time range: {trends_data.index.min()} to {trends_data.index.max()}")
    
    # Import feature engineering functions
    from .feature_engineer import create_supervised_dataset
    
    forecasts = {}
    trained_models = {}
    training_summary = {}
    
    successful_models = 0
    failed_models = 0
    
    # Process each topic
    for topic_name in trends_data.columns:
        print(f"\n{'='*60}")
        print(f"PROCESSING {topic_name}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Create supervised dataset for this topic
            topic_series = trends_data[topic_name]
            X, y = create_supervised_dataset(topic_series, look_back_window)
            
            if len(X) == 0:
                print(f"Skipping {topic_name}: insufficient data")
                failed_models += 1
                continue
            
            # Step 2: Train the model
            model, scalers, metrics, predictions = train_single_topic_model(
                (X, y), topic_name, look_back_window
            )
            
            if model is None:
                print(f"Failed to train model for {topic_name}")
                failed_models += 1
                continue
            
            # Step 3: Generate forecasts
            print(f"Generating {forecast_years}-year forecast...")
            
            # Get the last sequence from the time series for forecasting
            last_sequence = topic_series.values[-look_back_window:]
            
            future_forecasts = iterative_forecast(
                model, scalers, last_sequence, forecast_years
            )
            
            # Step 4: Store results
            forecasts[topic_name] = {
                'historical_data': topic_series.values,
                'historical_years': topic_series.index.tolist(),
                'forecasts': future_forecasts,
                'forecast_years': list(range(
                    trends_data.index.max() + 1, 
                    trends_data.index.max() + forecast_years + 1
                ))
            }
            
            trained_models[topic_name] = {
                'model': model,
                'scalers': scalers,
                'metrics': metrics,
                'predictions': predictions
            }
            
            training_summary[topic_name] = metrics
            
            print(f"Forecast for {topic_name}: {future_forecasts}")
            successful_models += 1
            
        except Exception as e:
            print(f"Error processing {topic_name}: {e}")
            failed_models += 1
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully trained models: {successful_models}")
    print(f"Failed models: {failed_models}")
    print(f"Success rate: {(successful_models/(successful_models + failed_models))*100:.1f}%")
    
    if successful_models > 0:
        # Calculate average performance metrics
        avg_rmse = np.mean([metrics['rmse'] for metrics in training_summary.values()])
        avg_r2 = np.mean([metrics['r2'] for metrics in training_summary.values()])
        
        print(f"\nAverage Model Performance:")
        print(f"  Average RMSE: {avg_rmse:.2f}")
        print(f"  Average R²: {avg_r2:.3f}")
        
        # Show top performing topics
        sorted_topics = sorted(training_summary.items(), key=lambda x: x[1]['r2'], reverse=True)
        print(f"\nTop 3 performing topics (by R²):")
        for i, (topic, metrics) in enumerate(sorted_topics[:3]):
            print(f"  {i+1}. {topic}: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.2f}")
    
    return forecasts, trained_models


def save_models(trained_models, save_directory="models"):
    """
    Save trained models to disk for later use.
    
    Args:
        trained_models (dict): Dictionary of trained models
        save_directory (str): Directory to save models
    """
    
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"Saving {len(trained_models)} models to {save_directory}/...")
    
    for topic_name, model_data in trained_models.items():
        try:
            # Save the model and scalers
            model_path = os.path.join(save_directory, f"{topic_name}_model.pkl")
            joblib.dump(model_data, model_path)
            print(f"Saved {topic_name} model")
        except Exception as e:
            print(f"Error saving {topic_name}: {e}")
    
    print("Model saving completed!")


def load_models(save_directory="models"):
    """
    Load previously saved models from disk.
    
    Args:
        save_directory (str): Directory containing saved models
        
    Returns:
        dict: Dictionary of loaded models
    """
    
    import os
    import glob
    
    model_files = glob.glob(os.path.join(save_directory, "*_model.pkl"))
    loaded_models = {}
    
    print(f"Loading models from {save_directory}/...")
    
    for model_file in model_files:
        try:
            topic_name = os.path.basename(model_file).replace("_model.pkl", "")
            model_data = joblib.load(model_file)
            loaded_models[topic_name] = model_data
            print(f"Loaded {topic_name} model")
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
    
    print(f"Successfully loaded {len(loaded_models)} models")
    
    return loaded_models


if __name__ == "__main__":
    # Test the model training module
    print("Testing model training module...")
    
    # Create sample time series data
    np.random.seed(42)
    years = list(range(2000, 2021))
    
    # Create sample trends data
    sample_trends = pd.DataFrame(
        index=years,
        data={
            'Topic_0': np.random.poisson(10, len(years)) + np.arange(len(years)) * 0.5,
            'Topic_1': np.random.poisson(8, len(years)) + np.arange(len(years)) * 0.3,
            'Topic_2': np.random.poisson(12, len(years)) + np.arange(len(years)) * 0.7
        }
    )
    
    try:
        # Test training
        forecasts, models = train_all_topics(sample_trends, look_back_window=3, forecast_years=3)
        
        print(f"\nTest completed successfully!")
        print(f"Generated forecasts for {len(forecasts)} topics")
        
        # Show sample forecast
        if forecasts:
            topic_name = list(forecasts.keys())[0]
            forecast_data = forecasts[topic_name]
            print(f"\nSample forecast for {topic_name}:")
            print(f"Forecast years: {forecast_data['forecast_years']}")
            print(f"Forecast values: {forecast_data['forecasts']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()