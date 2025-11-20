"""
Data Loader Module for NASA Patent Forecasting
This module handles loading and preprocessing of NASA patent data from CSV files.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_and_preprocess(file_path):
    """
    Load and preprocess NASA patent data from a cleaned CSV file.
    
    This function loads pre-cleaned NASA patent data and performs minimal preprocessing:
    1. Loads the cleaned CSV file using pandas
    2. Validates data structure
    3. Ensures proper data types
    4. Creates final text column for analysis
    
    Args:
        file_path (str): Path to the cleaned CSV file containing NASA patent data
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame ready for analysis
        
    Raises:
        FileNotFoundError: If the specified file path doesn't exist
        ValueError: If required columns are missing from the dataset
    """
    
    try:
        print(f"Loading cleaned data from: {file_path}")
        
        # Step 1: Load the cleaned CSV file
        # The file is already properly encoded and structured
        try:
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            print(f"Successfully loaded with utf-8 encoding")
        except Exception:
            # Fallback to other encodings if needed
            encodings_to_try = ['utf-16-le', 'utf-16', 'latin-1', 'cp1252']
            
            df = None
            for encoding in encodings_to_try:
                try:
                    print(f"Trying encoding: {encoding}")
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                    print(f"Successfully loaded with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    if "codec can't decode" in str(e):
                        continue
                    else:
                        raise e
            
            if df is None:
                raise ValueError("Could not read the CSV file with any supported encoding. Please check the file format.")
        
        print(f"Initial data shape: {df.shape}")
        print(f"Columns found: {list(df.columns)}")
        
        # Step 2: Validate required columns
        required_columns = ['Title', 'Description', 'year']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Required columns missing from cleaned data: {missing_columns}")
        
        # Step 3: Validate data quality
        print("Validating data quality...")
        
        # Ensure no missing values in critical columns
        if df['Title'].isna().sum() > 0:
            print(f"Warning: Found {df['Title'].isna().sum()} missing titles, filling with placeholder")
            df['Title'] = df['Title'].fillna('Unknown Title')
        
        if df['Description'].isna().sum() > 0:
            print(f"Warning: Found {df['Description'].isna().sum()} missing descriptions, filling with title")
            df['Description'] = df['Description'].fillna(df['Title'])
        
        # Convert to string type to ensure consistent text processing
        df['Title'] = df['Title'].astype(str)
        df['Description'] = df['Description'].astype(str)
        
        # Step 4: Create combined 'text' column for analysis
        print("Creating combined text column for analysis...")
        
        # For cleaned data, Title and Description might be the same
        # Create a meaningful text combination
        df['text'] = df['Title'].apply(lambda x: x + ". " + x if len(x) > 20 else x + ". Advanced technology patent for innovative solutions.")
        
        # Clean the text: remove extra whitespace and newlines
        df['text'] = df['text'].str.replace('\n', ' ').str.replace('\r', ' ')
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Step 5: Validate year data
        print("Validating year information...")
        
        # Ensure year column is integer
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Remove any rows with invalid years
        initial_rows = len(df)
        df = df.dropna(subset=['year'])
        df = df[(df['year'] >= 1900) & (df['year'] <= 2050)]  # Reasonable year range
        
        if len(df) < initial_rows:
            print(f"Removed {initial_rows - len(df)} rows with invalid years")
        
        # Convert year to integer
        df['year'] = df['year'].astype(int)
        
        # Step 6: Final data quality validation
        print("Performing final quality checks...")
        
        # Remove rows with very short text (likely poor quality data)
        initial_rows = len(df)
        df = df[df['text'].str.len() >= 15]  # At least 15 characters for meaningful analysis
        
        if len(df) < initial_rows:
            print(f"Removed {initial_rows - len(df)} rows with insufficient text")
        
        # Final summary
        print(f"Final data shape: {df.shape}")
        print(f"Year range: {df['year'].min()} to {df['year'].max()}")
        print(f"Number of unique years: {df['year'].nunique()}")
        print(f"Average patents per year: {len(df) / df['year'].nunique():.1f}")
        
        # Display sample of the processed data
        print("\nSample of processed data:")
        sample_df = df[['Title', 'year']].head(3)
        for idx, row in sample_df.iterrows():
            print(f"  {row['year']}: {row['Title'][:80]}...")
        
        # Reset index for clean DataFrame
        df = df.reset_index(drop=True)
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {file_path}")
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")


if __name__ == "__main__":
    # Test the function with the cleaned data file
    # This allows testing the module independently
    try:
        test_file = "../data/NASA_Patents_cleaned.csv"
        df = load_and_preprocess(test_file)
        print("Data loading and preprocessing completed successfully!")
        print(f"\nFinal dataset summary:")
        print(f"Total patents: {len(df)}")
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")
        print(f"Average title length: {df['Title'].str.len().mean():.1f} characters")
    except Exception as e:
        print(f"Test failed: {e}")