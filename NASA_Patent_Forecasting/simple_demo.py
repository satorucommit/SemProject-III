"""
Simple NASA Patent Forecasting Demo
This demonstrates the working patent analysis system.
"""

import sys
import os

# Add src directory to Python path
sys.path.append('src')

def simple_demo():
    """Simple demonstration of the patent forecasting system"""
    
    print("🚀 NASA PATENT TECHNOLOGY TREND FORECASTING SYSTEM 🚀")
    print("=" * 80)
    print("Welcome to the NASA Patent Forecasting Demo!")
    print("This system analyzes NASA patent data and evaluates new ideas.")
    print("=" * 80)
    
    try:
        # Step 1: Load and preprocess data
        print("\n📊 STEP 1: LOADING PATENT DATA")
        print("-" * 40)
        
        from src.data_loader import load_and_preprocess
        df = load_and_preprocess('data/NASA_Patents.csv')
        print(f"✅ Successfully loaded {len(df)} patent records")
        print(f"📅 Data spans from {df.year.min()} to {df.year.max()}")
        
        # Step 2: Discover technology topics
        print("\n🔍 STEP 2: DISCOVERING TECHNOLOGY TOPICS")
        print("-" * 40)
        
        from src.topic_modeler import discover_topics
        df_with_topics, vectorizer, nmf_model, topic_keywords = discover_topics(df, num_topics=8)
        print(f"✅ Successfully discovered {len(topic_keywords)} technology topics")
        
        # Display top topics
        print("\n🏷️  DISCOVERED TECHNOLOGY TOPICS:")
        for i in range(min(5, len(topic_keywords))):
            keywords = [word for word, score in topic_keywords[i][:4]]
            topic_count = len(df_with_topics[df_with_topics['topic'] == i])
            print(f"  Topic {i}: {', '.join(keywords)} ({topic_count} patents)")
        
        # Step 3: Analyze trends
        print("\n📈 STEP 3: ANALYZING TECHNOLOGY TRENDS")
        print("-" * 40)
        
        from src.feature_engineer import create_trends_table
        trends_table = create_trends_table(df_with_topics)
        print(f"✅ Created trends analysis table: {trends_table.shape}")
        
        # Show trend summary
        total_by_year = trends_table.sum(axis=1)
        print("📊 Patent trends by year:")
        for year, count in total_by_year.items():
            print(f"  {year}: {count} patents")
        
        # Test patent evaluation with forecasting
        print("\n💡 STEP 4: EVALUATING PATENT IDEAS WITH FORECASTING")
        print("-" * 40)
        
        # Create simple forecasts for evaluation
        from src.model_trainer_enhanced import enhanced_train_all_topics
        forecasts, models = enhanced_train_all_topics(trends_table, forecast_years=3)
        
        print(f"✅ Created forecasts for {len(forecasts)} topics")
        
        # Test patent ideas
        test_ideas = [
            "Advanced solar panel technology for spacecraft power generation",
            "Artificial intelligence navigation system for autonomous vehicles", 
            "Quantum communication system for secure data transmission"
        ]
        
        print("\nTesting patent idea evaluation with trend forecasting...")
        
        from src.trend_evaluator import evaluate_new_idea
        last_year = trends_table.index.max()
        
        for i, idea in enumerate(test_ideas):
            print(f"\n🔍 Idea {i+1}: '{idea[:50]}...'")
            try:
                # Full evaluation with forecasting
                result = evaluate_new_idea(idea, vectorizer, nmf_model, forecasts, last_year)
                
                # Extract key information from result
                lines = result.split('\n')
                for line in lines:
                    if 'Status:' in line or 'Explanation:' in line or 'Trend Classification:' in line:
                        print(f"   {line.strip()}")
                        if 'Explanation:' in line:
                            break
                            
            except Exception as e:
                print(f"   ❌ Evaluation failed: {e}")
                # Fallback to basic classification
                try:
                    predicted_topic, probabilities = predict_topic_for_new_idea(idea, vectorizer, nmf_model)
                    confidence = probabilities[predicted_topic]
                    topic_words = [word for word, score in topic_keywords[predicted_topic][:3]]
                    
                    print(f"   📌 Classified as: Topic {predicted_topic} ({', '.join(topic_words)})")
                    print(f"   🎯 Confidence: {confidence:.3f}")
                except Exception as e2:
                    print(f"   ❌ Classification also failed: {e2}")
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n💭 Key Features Demonstrated:")
        print("   ✅ Patent data loading and preprocessing")
        print("   ✅ Technology topic discovery using NLP")
        print("   ✅ Time-series trend analysis")
        print("   ✅ Patent idea classification and evaluation")
        print("\n🚀 To run the full interactive system: python main.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_demo():
    """Interactive patent evaluation demo"""
    
    print("\n" + "=" * 60)
    print("🔬 INTERACTIVE PATENT IDEA EVALUATOR")
    print("=" * 60)
    print("Enter your patent ideas and get instant technology trend analysis!")
    print("Type 'quit' to exit.")
    
    try:
        # Quick setup
        from src.data_loader import load_and_preprocess
        from src.topic_modeler import discover_topics
        from src.trend_evaluator import predict_topic_for_new_idea
        
        df = load_and_preprocess('data/NASA_Patents.csv')
        df_with_topics, vectorizer, nmf_model, topic_keywords = discover_topics(df, num_topics=8)
        
        print(f"\n✅ System ready! Loaded {len(df)} patents and {len(topic_keywords)} topics.")
        
        evaluation_count = 0
        
        while True:
            print("\n" + "-" * 60)
            user_input = input("🔍 Enter your patent idea: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if len(user_input) < 5:
                print("⚠️  Please provide a more detailed description.")
                continue
            
            try:
                print(f"\n🔄 Analyzing: '{user_input}'")
                
                predicted_topic, probabilities = predict_topic_for_new_idea(user_input, vectorizer, nmf_model)
                confidence = probabilities[predicted_topic]
                
                # Get topic keywords
                topic_words = [word for word, score in topic_keywords[predicted_topic][:4]]
                
                print("\n📊 ANALYSIS RESULTS:")
                print(f"   🎯 Technology Category: Topic {predicted_topic}")
                print(f"   🏷️  Keywords: {', '.join(topic_words)}")
                print(f"   📈 Classification Confidence: {confidence:.3f}")
                
                if confidence > 0.3:
                    print("   ✅ HIGH confidence classification")
                elif confidence > 0.15:
                    print("   ⚠️  MEDIUM confidence classification")
                else:
                    print("   ❌ LOW confidence - idea may be novel or unclear")
                
                evaluation_count += 1
                
            except Exception as e:
                print(f"❌ Analysis failed: {e}")
        
        print(f"\n👋 Thanks for using the demo! Total evaluations: {evaluation_count}")
        
    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")

if __name__ == "__main__":
    # Run simple demo first
    success = simple_demo()
    
    # If successful, offer interactive demo
    if success:
        response = input("\n🤔 Would you like to try the interactive evaluator? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_demo()
    
    print("\n🎉 Demo session completed!")