"""
Main Script for NASA Patent Forecasting Application
This script orchestrates the complete pipeline and provides user interaction.
"""

import sys
import os
import warnings
import time
from datetime import datetime

# Add src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all required modules from src directory
from src.data_loader import load_and_preprocess
from src.topic_modeler import discover_topics, display_topic_details
from src.feature_engineer import create_trends_table, analyze_trends_statistics

# Try to import TensorFlow version, fallback to enhanced version if needed
try:
    from src.model_trainer import train_all_topics
    print("✅ Using TensorFlow-enabled model trainer")
except ImportError as e:
    print(f"⚠️  TensorFlow import failed: {e}")
    print("🔄 Using enhanced model trainer (reliable forecasting)...")
    from src.model_trainer_enhanced import enhanced_train_all_topics as train_all_topics

from src.trend_evaluator import evaluate_new_idea

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration settings
CONFIG = {
    'data_file': '../data/NASA_Patents_cleaned.csv',
    'num_topics': 15,
    'look_back_window': 2,  # Reduced from 5 to work with limited historical data
    'forecast_years': 5,
    'test_size': 0.2
}


def print_header():
    """Print application header and welcome message."""
    
    print("="*80)
    print("🚀 NASA PATENT TECHNOLOGY TREND FORECASTING SYSTEM 🚀")
    print("="*80)
    print("Welcome to the NASA Patent Forecasting Application!")
    print("This system analyzes NASA patent data to predict technology trends")
    print("and evaluates new patent ideas against these predictions.")
    print()
    print("System Configuration:")
    print(f"  📊 Topic Discovery: {CONFIG['num_topics']} topics")
    print(f"  📈 Forecast Horizon: {CONFIG['forecast_years']} years")
    print(f"  🔄 Look-back Window: {CONFIG['look_back_window']} years")
    print("="*80)


def print_pipeline_stage(stage_number, stage_name, description):
    """Print formatted stage information."""
    
    print(f"\n{'='*60}")
    print(f"STAGE {stage_number}: {stage_name}")
    print(f"{'='*60}")
    print(description)
    print()


def load_data_stage():
    """
    Stage 1: Load and preprocess the NASA patent data.
    
    Returns:
        pd.DataFrame: Cleaned and preprocessed patent data
    """
    
    print_pipeline_stage(1, "DATA LOADING & PREPROCESSING", 
                         "Loading NASA patent data and preparing it for analysis...")
    
    try:
        # Load and preprocess the data
        df = load_and_preprocess(CONFIG['data_file'])
        
        print(f"✅ Successfully loaded {len(df)} patent records")
        print(f"📅 Data spans from {df['year'].min()} to {df['year'].max()}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file '{CONFIG['data_file']}'")
        print("Please ensure the NASA_Patents.csv file is in the data/ directory.")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


def topic_discovery_stage(df):
    """
    Stage 2: Discover technology topics using NLP.
    
    Args:
        df (pd.DataFrame): Preprocessed patent data
        
    Returns:
        tuple: (dataframe_with_topics, vectorizer, nmf_model, topic_keywords)
    """
    
    print_pipeline_stage(2, "TOPIC DISCOVERY", 
                         "Discovering technology topics using Natural Language Processing...")
    
    try:
        # Discover topics using TF-IDF and NMF
        df_with_topics, vectorizer, nmf_model, topic_keywords = discover_topics(
            df, num_topics=CONFIG['num_topics']
        )
        
        print(f"✅ Successfully discovered {CONFIG['num_topics']} technology topics")
        
        # Display topic summary
        print("\n🏷️  TOPIC SUMMARY:")
        display_topic_details(topic_keywords, num_topics_to_show=5)  # Show top 5 topics
        
        return df_with_topics, vectorizer, nmf_model, topic_keywords
        
    except Exception as e:
        print(f"❌ Error in topic discovery: {e}")
        sys.exit(1)


def trend_analysis_stage(df_with_topics):
    """
    Stage 3: Create time-series trends analysis.
    
    Args:
        df_with_topics (pd.DataFrame): Data with topic assignments
        
    Returns:
        pd.DataFrame: Time-series trends table
    """
    
    print_pipeline_stage(3, "TREND ANALYSIS", 
                         "Creating time-series trends analysis...")
    
    try:
        # Create trends table
        trends_table = create_trends_table(df_with_topics)
        
        print(f"✅ Created trends table: {trends_table.shape}")
        
        # Display trend statistics
        analyze_trends_statistics(trends_table)
        
        return trends_table
        
    except Exception as e:
        print(f"❌ Error in trend analysis: {e}")
        sys.exit(1)


def model_training_stage(trends_table):
    """
    Stage 4: Train forecasting models and generate predictions.
    
    Args:
        trends_table (pd.DataFrame): Time-series trends data
        
    Returns:
        tuple: (forecasts_dict, trained_models_dict)
    """
    
    print_pipeline_stage(4, "MODEL TRAINING & FORECASTING", 
                         "Training advanced ensemble models and generating forecasts...")
    
    try:
        # Train models and generate forecasts
        forecasts, trained_models = train_all_topics(
            trends_table,
            look_back_window=CONFIG['look_back_window'],
            forecast_years=CONFIG['forecast_years']
        )
        
        # Ensure we have forecasts available
        valid_forecasts = {k: v for k, v in forecasts.items() if v is not None}
        successful_models = len(valid_forecasts)
        
        print(f"✅ Successfully created {successful_models} forecasts")
        
        # If no forecasts were created, use enhanced fallback
        if successful_models == 0:
            print("⚠️  No forecasts created, using enhanced fallback...")
            from src.model_trainer_enhanced import enhanced_train_all_topics
            forecasts, trained_models = enhanced_train_all_topics(
                trends_table,
                look_back_window=CONFIG['look_back_window'],
                forecast_years=CONFIG['forecast_years']
            )
            valid_forecasts = forecasts
            successful_models = len(valid_forecasts)
        
        # Display sample forecasts
        print("\n📈 SAMPLE FORECASTS:")
        sample_topics = list(valid_forecasts.keys())[:3]  # Show first 3 topics
        
        for topic_name in sample_topics:
            if topic_name in valid_forecasts and valid_forecasts[topic_name] is not None:
                forecast_data = valid_forecasts[topic_name]
                print(f"\n{topic_name}:")
                for year, value in zip(forecast_data['forecast_years'][:3], forecast_data['forecasts'][:3]):
                    print(f"  {year}: {value:.1f} patents")
        
        return valid_forecasts, trained_models
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        sys.exit(1)


def interactive_evaluation_stage(vectorizer, nmf_model, forecasts, last_year):
    """
    Stage 5: Interactive patent idea evaluation.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        nmf_model: Trained NMF model
        forecasts (dict): Generated forecasts for all topics
        last_year (int): Last year of historical data
    """
    
    print_pipeline_stage(5, "INTERACTIVE EVALUATION", 
                         "Ready for patent idea evaluation! Enter your ideas below.")
    
    print("💡 HOW TO USE:")
    print("  • Enter your patent idea as text (e.g., 'solar panel efficiency technology')")
    print("  • The system will predict which technology topic it belongs to")
    print("  • You'll receive a detailed trend analysis and recommendation")
    print("  • Type 'exit' to quit the application")
    print("  • Type 'help' for more information")
    print("  • Type 'examples' to see sample ideas")
    
    evaluation_count = 0
    
    while True:
        try:
            print("\n" + "-"*60)
            user_input = input("🔍 Enter your patent idea (or 'exit' to quit): ").strip()
            
            # Handle special commands
            if user_input.lower() == 'exit':
                print("\n👋 Thank you for using the NASA Patent Forecasting System!")
                print(f"Total evaluations performed: {evaluation_count}")
                break
                
            elif user_input.lower() == 'help':
                show_help()
                continue
                
            elif user_input.lower() == 'examples':
                show_examples()
                continue
                
            elif not user_input:
                print("⚠️  Please enter a patent idea or 'exit' to quit.")
                continue
                
            elif len(user_input) < 10:
                print("⚠️  Please provide a more detailed description (at least 10 characters).")
                continue
            
            # Evaluate the patent idea
            print(f"\n🔄 Analyzing your idea: '{user_input}'")
            print("Please wait while we process your request...")
            
            start_time = time.time()
            
            # Perform the evaluation
            evaluation_result = evaluate_new_idea(
                user_input, vectorizer, nmf_model, forecasts, last_year
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Display the results
            print(f"\n{evaluation_result}")
            
            evaluation_count += 1
            print(f"\n⏱️  Analysis completed in {processing_time:.2f} seconds")
            print(f"📊 Total evaluations: {evaluation_count}")
            
            # Ask if user wants to save the result
            save_option = input("\n💾 Would you like to save this evaluation to a file? (y/n): ").strip().lower()
            if save_option in ['y', 'yes']:
                save_evaluation_result(user_input, evaluation_result, evaluation_count)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
            
        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            print("Please try again with a different input.")


def show_help():
    """Display help information for users."""
    
    print("\n" + "="*60)
    print("📖 HELP & USAGE GUIDE")
    print("="*60)
    print("This system evaluates patent ideas against NASA technology trends.")
    print()
    print("📝 INPUT GUIDELINES:")
    print("  • Provide a clear, descriptive patent idea")
    print("  • Include technical keywords related to your innovation")
    print("  • Minimum 10 characters for meaningful analysis")
    print("  • Examples: 'advanced battery technology', 'satellite communication'")
    print()
    print("📊 EVALUATION PROCESS:")
    print("  1. Your idea is classified into a technology topic")
    print("  2. The system retrieves forecasts for that topic")
    print("  3. Growth trends are analyzed and classified")
    print("  4. A recommendation is provided based on predicted trends")
    print()
    print("🏷️  TREND CLASSIFICATIONS:")
    print("  • HIGH-GROWTH (Booming): >20% growth expected")
    print("  • MODERATE-GROWTH (Growing): 10-20% growth expected")
    print("  • LOW-GROWTH (Slow Growth): 5-10% growth expected")
    print("  • STABLE (Steady): -5% to 5% change expected")
    print("  • DECLINING (Decreasing): <-5% change expected")
    print()
    print("💡 SPECIAL COMMANDS:")
    print("  • 'help' - Show this help message")
    print("  • 'examples' - Show example patent ideas")
    print("  • 'exit' - Quit the application")


def show_examples():
    """Display example patent ideas for users."""
    
    print("\n" + "="*60)
    print("💡 EXAMPLE PATENT IDEAS")
    print("="*60)
    print("Here are some example patent ideas you can try:")
    print()
    
    examples = [
        "Advanced solar panel efficiency technology for spacecraft",
        "Artificial intelligence system for autonomous spacecraft navigation",
        "Lightweight composite materials for rocket construction",
        "Advanced life support systems for long-duration space missions",
        "Quantum communication technology for satellite networks",
        "Advanced propulsion system using ion drive technology",
        "Smart sensors for monitoring spacecraft health",
        "Advanced thermal protection systems for atmospheric re-entry",
        "Robotic systems for automated space station maintenance",
        "Advanced water recycling technology for space habitats"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"  {i:2d}. {example}")
    
    print("\n💭 You can copy any of these examples or create your own!")
    print("   The more specific and technical your description, the better the analysis.")


def save_evaluation_result(idea_text, evaluation_result, evaluation_number):
    """Save evaluation result to a file."""
    
    try:
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/evaluation_{evaluation_number}_{timestamp}.txt"
        
        # Save the result
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"NASA Patent Forecasting - Evaluation Result\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Evaluation #{evaluation_number}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Patent Idea: {idea_text}\n\n")
            f.write(evaluation_result)
        
        print(f"✅ Evaluation saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error saving evaluation: {e}")


def main():
    """
    Main function that orchestrates the complete NASA Patent Forecasting pipeline.
    
    This function executes the following stages:
    1. Data Loading & Preprocessing
    2. Topic Discovery using NLP
    3. Time-series Trend Analysis
    4. Model Training & Forecasting
    5. Interactive Patent Idea Evaluation
    """
    
    # Print welcome message
    print_header()
    
    start_time = time.time()
    
    try:
        # Stage 1: Load and preprocess data
        df = load_data_stage()
        
        # Stage 2: Discover topics
        df_with_topics, vectorizer, nmf_model, topic_keywords = topic_discovery_stage(df)
        
        # Stage 3: Create trends analysis
        trends_table = trend_analysis_stage(df_with_topics)
        
        # Stage 4: Train models and generate forecasts
        forecasts, trained_models = model_training_stage(trends_table)
        
        # Calculate pipeline completion time
        pipeline_time = time.time() - start_time
        last_historical_year = trends_table.index.max()
        
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total processing time: {pipeline_time:.2f} seconds")
        print(f"📊 Ready to evaluate patent ideas against {len(forecasts)} technology trends")
        
        # Stage 5: Interactive evaluation
        interactive_evaluation_stage(vectorizer, nmf_model, forecasts, last_historical_year)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user. Exiting...")
        
    except Exception as e:
        print(f"\n❌ Critical error in pipeline: {e}")
        print("Please check your data file and try again.")
        import traceback
        traceback.print_exc()
        
    finally:
        total_time = time.time() - start_time
        print(f"\n⏱️  Total session time: {total_time:.2f} seconds")
        print("Goodbye! 👋")


if __name__ == "__main__":
    main()