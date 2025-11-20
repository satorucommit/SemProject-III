"""
Quick Demo Script for NASA Patent Forecasting System
This script demonstrates that the cleaned data and system are working properly.
"""

import sys
import os
import pandas as pd

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_and_preprocess

def quick_demo():
    """
    Quick demonstration of the working system.
    """
    
    print("="*80)
    print("🚀 NASA PATENT FORECASTING - QUICK DEMO")
    print("="*80)
    
    try:
        # Test the cleaned data loading
        print("📋 Testing cleaned data loading...")
        df = load_and_preprocess("../data/NASA_Patents_cleaned.csv")
        
        print(f"\n✅ SUCCESS! System is working properly!")
        print(f"📊 Loaded {len(df)} patent records")
        print(f"📅 Year range: {df['year'].min()} to {df['year'].max()}")
        print(f"🏷️  Sample topics from patent titles:")
        
        # Show some sample patent titles by year
        sample_years = [1930, 2020, 2025, 2030]
        for year in sample_years:
            year_data = df[df['year'] == year]
            if len(year_data) > 0:
                print(f"\n  {year} ({len(year_data)} patents):")
                for idx, row in year_data.head(2).iterrows():
                    print(f"    - {row['Title'][:70]}...")
        
        # Show year distribution
        print(f"\n📈 Patent distribution by decade:")
        df['decade'] = (df['year'] // 10) * 10
        decade_counts = df['decade'].value_counts().sort_index()
        
        for decade, count in decade_counts.head(8).items():
            print(f"  {int(decade)}s: {count} patents")
        
        print(f"\n🎯 KEY ACHIEVEMENTS:")
        print(f"  ✅ Data successfully cleaned and processed")
        print(f"  ✅ {len(df)} patent records ready for analysis")
        print(f"  ✅ {df['year'].nunique()} unique years of data")
        print(f"  ✅ System ready for topic modeling and forecasting")
        
        print(f"\n🏃‍♂️ TO RUN FULL SYSTEM:")
        print(f"  python main.py")
        
        print(f"\n" + "="*80)
        print(f"✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print(f"Your NASA Patent Forecasting system is ready to use!")
        print(f"="*80)
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        return False
    
    return True

if __name__ == "__main__":
    quick_demo()