"""
Feature Engineering Module for NASA Patent Forecasting
This module handles the creation of time-series trends data and supervised learning datasets.
"""

import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def create_trends_table(dataframe):
    """
    Create a time-series trends table from patent data with topic assignments.
    
    This function transforms the patent data into a time-series format where:
    - Rows represent years
    - Columns represent different technology topics
    - Values represent the count of patents for each topic in each year
    
    Args:
        dataframe (pd.DataFrame): DataFrame with 'year' and 'topic' columns
        
    Returns:
        pd.DataFrame: Time-series DataFrame with years as index and topics as columns
        
    Example:
        Input: DataFrame with patents assigned to topics by year
        Output: 
                Topic_0  Topic_1  Topic_2  ...
        Year
        2000        5       12        8   ...
        2001        7       15       10   ...
        2002        6       18       12   ...
    """
    
    print("Creating time-series trends table...")
    
    # Step 1: Validate input data
    required_columns = ['year', 'topic']
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")
    
    # Remove any rows with missing year or topic data
    clean_df = dataframe.dropna(subset=['year', 'topic']).copy()
    
    print(f"Processing {len(clean_df)} patents across {clean_df['year'].nunique()} years")
    print(f"Year range: {clean_df['year'].min()} to {clean_df['year'].max()}")
    print(f"Number of topics: {clean_df['topic'].nunique()}")
    
    # Step 2: Create pivot table
    # This aggregates the data by counting patents for each year-topic combination
    # We need to create a count column first since pivot_table expects values to aggregate
    clean_df['patent_count'] = 1  # Each row represents one patent
    
    trends_table = pd.pivot_table(
        clean_df,
        values='patent_count',        # Count the number of patents
        index='year',                 # Years as rows
        columns='topic',              # Topics as columns
        aggfunc='sum',                # Sum the counts (each patent contributes 1)
        fill_value=0                  # Fill missing combinations with 0
    )
    
    # Step 3: Ensure consistent column naming
    # Rename columns to have consistent Topic_X format
    trends_table.columns = [f'Topic_{int(col)}' for col in trends_table.columns]
    
    # Step 4: Ensure complete year coverage
    # Fill any missing years with zeros to create a complete time series
    year_min = int(trends_table.index.min())
    year_max = int(trends_table.index.max())
    
    # Create complete year range
    complete_years = pd.Index(range(year_min, year_max + 1), name='year')
    
    # Reindex to include all years, filling missing years with 0
    trends_table = trends_table.reindex(complete_years, fill_value=0)
    
    # Step 5: Data quality checks and statistics
    print(f"\nTrends table created successfully!")
    print(f"Shape: {trends_table.shape}")
    print(f"Years covered: {len(trends_table)} years ({year_min} to {year_max})")
    print(f"Topics covered: {len(trends_table.columns)} topics")
    
    # Calculate and display basic statistics
    total_patents_per_year = trends_table.sum(axis=1)
    print(f"Average patents per year: {total_patents_per_year.mean():.1f}")
    print(f"Year with most patents: {total_patents_per_year.idxmax()} ({total_patents_per_year.max()} patents)")
    print(f"Year with least patents: {total_patents_per_year.idxmin()} ({total_patents_per_year.min()} patents)")
    
    # Show most and least active topics
    total_patents_per_topic = trends_table.sum(axis=0)
    most_active_topic = total_patents_per_topic.idxmax()
    least_active_topic = total_patents_per_topic.idxmin()
    
    print(f"Most active topic: {most_active_topic} ({total_patents_per_topic.max()} total patents)")
    print(f"Least active topic: {least_active_topic} ({total_patents_per_topic.min()} total patents)")
    
    # Display sample of the trends table
    print(f"\nSample of trends table (first 5 years, first 5 topics):")
    print(trends_table.iloc[:5, :5])
    
    return trends_table


def create_supervised_dataset(time_series_data, look_back=5):
    """
    Transform time series data into supervised learning format using sliding window technique.
    
    This function creates training examples where:
    - X (features): Patent counts for the previous 'look_back' years
    - y (target): Patent count for the next year
    
    The sliding window approach allows the model to learn patterns from historical data
    to predict future values.
    
    Args:
        time_series_data (pd.Series or list): Time series of patent counts for a single topic
        look_back (int): Number of previous years to use as features (default: 5)
        
    Returns:
        tuple: (X, y) where X is features array and y is targets array
        
    Example:
        If time_series = [10, 12, 15, 18, 20, 22, 25] and look_back = 3:
        X = [[10, 12, 15],    y = [18,
             [12, 15, 18],          20,
             [15, 18, 20],          22,
             [18, 20, 22]]          25]
    """
    
    print(f"Creating supervised dataset with look_back window of {look_back} years...")
    
    # Step 1: Convert input to numpy array for easier manipulation
    if isinstance(time_series_data, pd.Series):
        data = time_series_data.values
    else:
        data = np.array(time_series_data)
    
    print(f"Time series length: {len(data)} data points")
    
    # Step 2: Validate input
    if len(data) < look_back + 1:
        raise ValueError(f"Time series too short. Need at least {look_back + 1} data points, got {len(data)}")
    
    # Step 3: Create sliding windows
    X, y = [], []
    
    # Iterate through the data creating sliding windows
    for i in range(look_back, len(data)):
        # Features: previous 'look_back' values
        X.append(data[i - look_back:i])
        # Target: current value (what we want to predict)
        y.append(data[i])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} training examples")
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    
    # Step 4: Display sample of created dataset
    if len(X) > 0:
        print(f"\nSample training examples:")
        num_samples_to_show = min(3, len(X))
        for i in range(num_samples_to_show):
            features = ', '.join([f'{val:.0f}' for val in X[i]])
            target = y[i]
            print(f"Example {i+1}: Features=[{features}] -> Target={target:.0f}")
    
    return X, y


def create_all_supervised_datasets(trends_table, look_back=5):
    """
    Create supervised datasets for all topics in the trends table.
    
    Args:
        trends_table (pd.DataFrame): Time-series DataFrame with topics as columns
        look_back (int): Number of previous years to use as features
        
    Returns:
        dict: Dictionary mapping topic names to (X, y) tuples
    """
    
    print(f"Creating supervised datasets for all {len(trends_table.columns)} topics...")
    
    supervised_datasets = {}
    
    for topic_name in trends_table.columns:
        print(f"\nProcessing {topic_name}...")
        
        # Get time series for this topic
        topic_series = trends_table[topic_name]
        
        try:
            # Create supervised dataset
            X, y = create_supervised_dataset(topic_series, look_back)
            supervised_datasets[topic_name] = (X, y)
            
            print(f"Successfully created dataset for {topic_name}: {len(X)} examples")
            
        except ValueError as e:
            print(f"Warning: Could not create dataset for {topic_name}: {e}")
            # Store empty arrays for topics with insufficient data
            supervised_datasets[topic_name] = (np.array([]), np.array([]))
    
    # Summary statistics
    valid_datasets = [name for name, (X, y) in supervised_datasets.items() if len(X) > 0]
    print(f"\nSuccessfully created datasets for {len(valid_datasets)} out of {len(trends_table.columns)} topics")
    
    if valid_datasets:
        avg_examples = np.mean([len(supervised_datasets[name][0]) for name in valid_datasets])
        print(f"Average training examples per topic: {avg_examples:.1f}")
    
    return supervised_datasets


def analyze_trends_statistics(trends_table):
    """
    Analyze and display statistical information about the trends data.
    
    Args:
        trends_table (pd.DataFrame): Time-series DataFrame with topics as columns
    """
    
    print("\n" + "="*80)
    print("TRENDS ANALYSIS REPORT")
    print("="*80)
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"Time period: {trends_table.index.min()} to {trends_table.index.max()}")
    print(f"Number of years: {len(trends_table)}")
    print(f"Number of topics: {len(trends_table.columns)}")
    print(f"Total patents: {trends_table.sum().sum()}")
    
    # Year-wise statistics
    print(f"\nYearly Statistics:")
    yearly_totals = trends_table.sum(axis=1)
    print(f"Average patents per year: {yearly_totals.mean():.1f}")
    print(f"Standard deviation: {yearly_totals.std():.1f}")
    print(f"Minimum year total: {yearly_totals.min()} (Year: {yearly_totals.idxmin()})")
    print(f"Maximum year total: {yearly_totals.max()} (Year: {yearly_totals.idxmax()})")
    
    # Topic-wise statistics
    print(f"\nTopic Statistics:")
    topic_totals = trends_table.sum(axis=0)
    print(f"Average patents per topic: {topic_totals.mean():.1f}")
    print(f"Standard deviation: {topic_totals.std():.1f}")
    print(f"Most active topic: {topic_totals.idxmax()} ({topic_totals.max()} patents)")
    print(f"Least active topic: {topic_totals.idxmin()} ({topic_totals.min()} patents)")
    
    # Growth trends
    print(f"\nGrowth Trends:")
    first_half = trends_table.iloc[:len(trends_table)//2].sum(axis=1).mean()
    second_half = trends_table.iloc[len(trends_table)//2:].sum(axis=1).mean()
    growth_rate = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
    
    print(f"Average patents (first half): {first_half:.1f}")
    print(f"Average patents (second half): {second_half:.1f}")
    print(f"Overall growth rate: {growth_rate:.1f}%")


if __name__ == "__main__":
    # Test the feature engineering functions
    print("Testing feature engineering module...")
    
    # Create sample data
    np.random.seed(42)
    
    # Simulate patent data
    years = list(range(2000, 2021))  # 21 years
    topics = [0, 1, 2]  # 3 topics
    
    sample_data = []
    for year in years:
        for topic in topics:
            # Simulate varying patent counts with some trend
            base_count = 5 + topic * 2  # Different base levels for topics
            trend = (year - 2000) * 0.5  # Slight upward trend
            noise = np.random.poisson(2)  # Random variation
            count = max(1, int(base_count + trend + noise))
            
            # Add individual patent records
            for _ in range(count):
                sample_data.append({'year': year, 'topic': topic})
    
    df = pd.DataFrame(sample_data)
    
    try:
        # Test create_trends_table
        print("\n1. Testing create_trends_table...")
        trends_table = create_trends_table(df)
        
        # Test create_supervised_dataset
        print("\n2. Testing create_supervised_dataset...")
        topic_0_series = trends_table['Topic_0']
        X, y = create_supervised_dataset(topic_0_series, look_back=3)
        
        # Test create_all_supervised_datasets
        print("\n3. Testing create_all_supervised_datasets...")
        all_datasets = create_all_supervised_datasets(trends_table, look_back=3)
        
        # Test analyze_trends_statistics
        print("\n4. Testing analyze_trends_statistics...")
        analyze_trends_statistics(trends_table)
        
        print("\nFeature engineering test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()