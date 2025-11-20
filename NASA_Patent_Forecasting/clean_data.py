"""
Data Cleaning Script for NASA Patent Forecasting
This script processes the NASA patent CSV file to extract only Title and Year information.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def parse_date_to_year(date_str):
    """
    Parse various date formats to extract year.
    
    Args:
        date_str: String representation of date
        
    Returns:
        int: Year or None if parsing fails
    """
    if pd.isna(date_str) or not date_str or str(date_str).strip() == '':
        return None
    
    date_str = str(date_str).strip()
    
    # Try different date parsing approaches
    try:
        # Common patterns in the data
        patterns = [
            r'(\d{2})\s+(\w+)\s+(\d{4})',  # "02 December 2024"
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # "12/24/2018"
            r'(\d{1,2})-(\d{1,2})-(\d{4})',  # "12-24-2018"
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # "2018-12-24"
            r'(\d{4})/(\d{1,2})/(\d{1,2})',  # "2018/12/24"
            r'(\d{4})',  # just year "2024"
        ]
        
        # Pattern for "02 December 2024" format
        month_match = re.search(r'\d{1,2}\s+\w+\s+(\d{4})', date_str)
        if month_match:
            return int(month_match.group(1))
        
        # Pattern for dates with year at the end
        year_match = re.search(r'\b(\d{4})\b', date_str)
        if year_match:
            year = int(year_match.group(1))
            # Check if year is reasonable
            if 1900 <= year <= 2100:
                return year
        
        # Try pandas parsing as fallback
        parsed_date = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(parsed_date):
            return parsed_date.year
            
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
    
    return None


def clean_title(title):
    """
    Clean and standardize patent titles.
    
    Args:
        title: Raw title string
        
    Returns:
        str: Cleaned title
    """
    if pd.isna(title) or not title:
        return "Unknown Patent Title"
    
    title = str(title).strip()
    
    # Remove extra whitespace and normalize
    title = re.sub(r'\s+', ' ', title)
    
    # Remove common artifacts
    title = title.replace('\n', ' ').replace('\r', ' ')
    
    # Ensure title is not empty after cleaning
    if not title or len(title) < 3:
        return "Unknown Patent Title"
    
    return title


def clean_nasa_patent_data(input_file, output_file):
    """
    Clean NASA patent data by extracting only Title and Year information.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output cleaned CSV file
    """
    
    print("="*80)
    print("🧹 NASA PATENT DATA CLEANING PROCESS")
    print("="*80)
    
    try:
        # Step 1: Read the original CSV file
        print(f"📖 Reading data from: {input_file}")
        
        # Try multiple encodings
        encodings = ['utf-8', 'utf-16-le', 'utf-16', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                print(f"   Trying encoding: {encoding}")
                df = pd.read_csv(input_file, encoding=encoding, low_memory=False)
                print(f"   ✅ Successfully loaded with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if "codec can't decode" in str(e):
                    continue
                else:
                    raise e
        
        if df is None:
            raise ValueError("Could not read the CSV file with any supported encoding")
        
        print(f"📊 Original data shape: {df.shape}")
        print(f"📋 Original columns: {list(df.columns)}")
        
        # Step 2: Extract required columns
        print("\n🔍 Extracting required columns...")
        
        # Check for required columns
        required_columns = {
            'Title': ['Title', 'title', 'TITLE'],
            'Patent Expiration Date': ['Patent Expiration Date', 'Expiration Date', 'expiration_date', 'Date', 'date']
        }
        
        column_mapping = {}
        
        for target_col, possible_names in required_columns.items():
            found = False
            for possible_name in possible_names:
                if possible_name in df.columns:
                    column_mapping[possible_name] = target_col
                    found = True
                    break
            
            if not found:
                raise ValueError(f"Could not find column for {target_col}. Available columns: {list(df.columns)}")
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        print(f"   ✅ Found required columns: {list(column_mapping.keys())}")
        
        # Step 3: Clean Title column
        print("\n🧽 Cleaning title data...")
        
        initial_rows = len(df)
        
        # Clean titles
        df['Title'] = df['Title'].apply(clean_title)
        
        # Remove rows with very short or generic titles
        df = df[df['Title'].str.len() >= 10]
        df = df[~df['Title'].str.contains('Unknown Patent Title')]
        
        print(f"   📉 Removed {initial_rows - len(df)} rows with poor title quality")
        
        # Step 4: Extract year from Patent Expiration Date
        print("\n📅 Extracting year information...")
        
        # Parse years from expiration dates
        df['year'] = df['Patent Expiration Date'].apply(parse_date_to_year)
        
        # Remove rows without valid years
        initial_rows = len(df)
        df = df.dropna(subset=['year'])
        df = df[df['year'].between(1900, 2100)]
        
        print(f"   📉 Removed {initial_rows - len(df)} rows with invalid years")
        print(f"   📊 Year range: {int(df['year'].min())} to {int(df['year'].max())}")
        
        # Step 5: Create final cleaned dataset
        print("\n🏗️  Creating final dataset...")
        
        # Keep only required columns
        cleaned_df = pd.DataFrame({
            'Title': df['Title'],
            'Description': df['Title'],  # Use title as description since no separate description exists
            'Expiration Date': df['year'].astype(int).astype(str) + '-01-01',  # Convert year to date format
            'year': df['year'].astype(int)
        })
        
        # Sort by year for better organization
        cleaned_df = cleaned_df.sort_values('year').reset_index(drop=True)
        
        print(f"   📊 Final dataset shape: {cleaned_df.shape}")
        print(f"   📈 Patents per year distribution:")
        
        # Show distribution by year
        year_counts = cleaned_df['year'].value_counts().sort_index()
        print(f"      Years covered: {len(year_counts)} years")
        print(f"      Average patents per year: {year_counts.mean():.1f}")
        print(f"      Year with most patents: {year_counts.idxmax()} ({year_counts.max()} patents)")
        
        # Step 6: Save cleaned data
        print(f"\n💾 Saving cleaned data to: {output_file}")
        
        cleaned_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print("✅ Data cleaning completed successfully!")
        
        # Step 7: Display sample of cleaned data
        print("\n📋 Sample of cleaned data:")
        print(cleaned_df.head(10).to_string(index=False))
        
        # Step 8: Data quality summary
        print(f"\n📈 CLEANING SUMMARY:")
        print(f"   Original records: {initial_rows}")
        print(f"   Final records: {len(cleaned_df)}")
        print(f"   Data retention: {len(cleaned_df)/initial_rows*100:.1f}%")
        print(f"   Years covered: {int(cleaned_df['year'].min())} - {int(cleaned_df['year'].max())}")
        print(f"   Average title length: {cleaned_df['Title'].str.len().mean():.1f} characters")
        
        return cleaned_df
        
    except Exception as e:
        print(f"❌ Error during data cleaning: {e}")
        raise


if __name__ == "__main__":
    # Define file paths
    input_file = "../data/NASA_Patents.csv"
    output_file = "../data/NASA_Patents_cleaned.csv"
    
    try:
        # Clean the data
        cleaned_data = clean_nasa_patent_data(input_file, output_file)
        
        print(f"\n🎉 Process completed! Cleaned data saved to: {output_file}")
        print("You can now run the main forecasting application with the cleaned data.")
        
    except Exception as e:
        print(f"\n❌ Failed to clean data: {e}")
        import traceback
        traceback.print_exc()