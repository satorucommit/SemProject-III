"""
Quick test script to demonstrate the NASA Patent Forecasting system
"""

import sys
import os

# Add src directory to Python path
sys.path.append('src')

def test_patent_evaluation():
    """Test the patent evaluation functionality"""
    
    print("🚀 NASA Patent Forecasting - Quick Demo")
    print("=" * 50)
    
    try:
        # Test data loading
        from src.data_loader import load_and_preprocess
        print("📊 Loading patent data...")
        df = load_and_preprocess('data/NASA_Patents.csv')
        print(f"✅ Loaded {len(df)} patents from {df.year.min()} to {df.year.max()}")
        
        # Test topic discovery
        from src.topic_modeler import discover_topics
        print("\n🔍 Discovering technology topics...")
        df_with_topics, vectorizer, nmf_model, topic_keywords = discover_topics(df, num_topics=10)
        print(f"✅ Discovered {len(topic_keywords)} topics")
        
        # Show top 3 topics
        print("\n🏷️  Top 3 Technology Topics:")
        for i in range(min(3, len(topic_keywords))):
            keywords = [word for word, score in topic_keywords[i][:5]]
            print(f"  Topic {i}: {', '.join(keywords)}")
        
        # Test trend analysis
        from src.feature_engineer import create_trends_table
        print("\n📈 Creating trends analysis...")
        trends_table = create_trends_table(df_with_topics)
        print(f"✅ Created trends table: {trends_table.shape}")
        
        # Try model training with reduced look-back
        print("\n🤖 Testing model training...")
        try:
            from src.model_trainer_fallback import train_all_topics_fallback
            forecasts, models = train_all_topics_fallback(trends_table, look_back_window=2, forecast_years=3)
            print(f"✅ Trained {len(models)} models successfully")
            
            if len(forecasts) > 0:
                # Test patent evaluation
                print("\n💡 Testing patent idea evaluation...")
                from src.trend_evaluator import evaluate_new_idea
                
                test_ideas = [
                    "Advanced solar panel technology for spacecraft",
                    "Artificial intelligence navigation system",
                    "Quantum communication array for space"
                ]
                
                last_year = trends_table.index.max()
                
                for i, idea in enumerate(test_ideas[:2]):  # Test first 2 ideas
                    print(f"\n🔍 Testing idea {i+1}: '{idea}'")
                    try:
                        result = evaluate_new_idea(idea, vectorizer, nmf_model, forecasts, last_year)
                        print("✅ Evaluation completed successfully!")
                        # Show just the recommendation part
                        lines = result.split('\n')
                        for line in lines:
                            if '💡 RECOMMENDATION:' in line or 'Status:' in line or 'Explanation:' in line:
                                print(f"   {line}")
                                if 'Explanation:' in line:
                                    break
                    except Exception as e:
                        print(f"⚠️  Evaluation failed: {e}")
            
        except Exception as e:
            print(f"⚠️  Model training failed: {e}")
            print("Using basic topic classification only...")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💭 To run the full interactive system:")
        print("   python main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_patent_evaluation()