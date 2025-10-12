"""
Test script to verify TensorFlow and NASA Patent Forecasting setup
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("🔍 Testing all imports...")
    
    try:
        # Core libraries
        import pandas as pd
        import numpy as np
        print("✅ Core libraries (pandas, numpy) - OK")
        
        # Machine learning
        import sklearn
        import xgboost as xgb
        print("✅ ML libraries (sklearn, xgboost) - OK")
        
        # TensorFlow
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} - OK")
        
        # Keras and scikeras
        from tensorflow import keras
        import scikeras
        print("✅ Keras and SciKeras - OK")
        
        # Project modules
        sys.path.append('src')
        from src.data_loader import load_and_preprocess
        from src.topic_modeler import discover_topics
        from src.feature_engineer import create_trends_table
        from src.model_trainer import train_all_topics
        from src.trend_evaluator import evaluate_new_idea
        print("✅ All project modules - OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_tensorflow_basic():
    """Test basic TensorFlow functionality"""
    print("\n🔍 Testing TensorFlow basic functionality...")
    
    try:
        import tensorflow as tf
        
        # Test tensor creation
        x = tf.constant([1, 2, 3, 4])
        y = tf.square(x)
        print(f"✅ Basic tensor operations - OK: {y.numpy()}")
        
        # Test simple model creation
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        model.compile(optimizer='adam', loss='mse')
        print("✅ Model creation and compilation - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def test_project_structure():
    """Test project file structure"""
    print("\n🔍 Testing project structure...")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'data/NASA_Patents.csv',
        'src/data_loader.py',
        'src/topic_modeler.py',
        'src/feature_engineer.py',
        'src/model_trainer.py',
        'src/model_trainer_fallback.py',
        'src/trend_evaluator.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present - OK")
        return True

def main():
    """Run all tests"""
    print("🚀 NASA Patent Forecasting - System Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test project structure
    if not test_project_structure():
        all_tests_passed = False
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test TensorFlow
    if not test_tensorflow_basic():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests PASSED! Your system is ready to run!")
        print("\n💡 To run the main application:")
        print("   python main.py")
        print("\n💡 The system will automatically use:")
        print("   - TensorFlow-enabled models (primary)")
        print("   - Fallback models if TensorFlow fails (backup)")
    else:
        print("⚠️  Some tests FAILED. Please check the errors above.")
    
    return all_tests_passed

if __name__ == "__main__":
    main()