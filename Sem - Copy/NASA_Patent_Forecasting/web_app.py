"""
Web Interface for NASA Patent Forecasting System
This module provides a simple web interface to interact with the NASA Patent Forecasting functionality.
"""

import sys
import os
import json
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import required modules
from src.data_loader import load_and_preprocess
from src.topic_modeler import discover_topics, display_topic_details
from src.feature_engineer import create_trends_table, analyze_trends_statistics
from src.model_trainer_enhanced import enhanced_train_all_topics
from src.trend_evaluator import evaluate_new_idea

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # Enable CORS for frontend integration

# Global variables to store trained models and data
vectorizer = None
nmf_model = None
forecasts = None
last_historical_year = None
topic_keywords = None
system_initialized = False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'NASA Patent Forecasting Web Interface'
    })

@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize the forecasting system"""
    global vectorizer, nmf_model, forecasts, last_historical_year, topic_keywords, system_initialized
    
    try:
        logger.info("Initializing NASA Patent Forecasting system...")
        
        # Load and preprocess data
        data_file = 'data/NASA_Patents_cleaned.csv'
        num_topics = 15
        look_back_window = 2
        forecast_years = 5
        
        df = load_and_preprocess(data_file)
        logger.info(f"Loaded {len(df)} patent records")
        
        # Discover topics
        df_with_topics, vectorizer, nmf_model, topic_keywords = discover_topics(df, num_topics=num_topics)
        logger.info(f"Discovered {num_topics} technology topics")
        
        # Create trends table
        trends_table = create_trends_table(df_with_topics)
        logger.info(f"Created trends table with shape {trends_table.shape}")
        
        # Train models and generate forecasts
        forecasts, trained_models = enhanced_train_all_topics(
            trends_table,
            look_back_window=look_back_window,
            forecast_years=forecast_years
        )
        logger.info(f"Generated forecasts for {len(forecasts)} topics")
        
        last_historical_year = trends_table.index.max()
        system_initialized = True
        
        return jsonify({
            'status': 'success',
            'message': 'System initialized successfully',
            'topics': list(forecasts.keys()),
            'year_range': f"{trends_table.index.min()}-{trends_table.index.max()}",
            'num_patents': len(df),
            'num_topics': num_topics
        })
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_idea():
    """Evaluate a patent idea"""
    global vectorizer, nmf_model, forecasts, last_historical_year, system_initialized, topic_keywords
    
    if not system_initialized:
        return jsonify({
            'status': 'error', 
            'message': 'System not initialized. Please initialize the system first.'
        }), 400
    
    try:
        # Handle case where request.json might be None
        request_data = request.get_json() or {}
        idea_text = request_data.get('idea', '')
        
        if not idea_text:
            return jsonify({'status': 'error', 'message': 'No idea text provided'}), 400
        
        logger.info(f"Evaluating patent idea: {idea_text[:50]}...")
        
        # Evaluate the idea
        evaluation_result = evaluate_new_idea(
            idea_text, vectorizer, nmf_model, forecasts, last_historical_year
        )
        
        # Parse the evaluation result to extract structured data
        parsed_result = parse_evaluation_result(evaluation_result, idea_text)
        
        return jsonify({
            'status': 'success',
            'idea': idea_text,
            'evaluation': evaluation_result,
            'parsed_result': parsed_result
        })
    except Exception as e:
        logger.error(f"Error evaluating idea: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def parse_evaluation_result(evaluation_text, idea_text):
    """Parse the evaluation text to extract structured data"""
    # Extract key information using regex
    result = {
        'idea': idea_text,
        'topic_classification': {},
        'trend_analysis': {},
        'recommendation': {},
        'additional_insights': []
    }
    
    lines = evaluation_text.split('\n')
    
    # Extract topic classification
    for line in lines:
        if 'Predicted Topic:' in line:
            match = re.search(r'Predicted Topic:\s*(.+)', line)
            if match:
                result['topic_classification']['predicted_topic'] = match.group(1).strip()
        elif 'Classification Confidence:' in line:
            match = re.search(r'Classification Confidence:\s*(.+)', line)
            if match:
                result['topic_classification']['confidence'] = match.group(1).strip()
        elif 'Topic Probability:' in line:
            match = re.search(r'Topic Probability:\s*([\d.]+)', line)
            if match:
                result['topic_classification']['probability'] = float(match.group(1))
    
    # Extract trend analysis
    for line in lines:
        if 'Trend Classification:' in line:
            match = re.search(r'Trend Classification:\s*(.+)', line)
            if match:
                result['trend_analysis']['classification'] = match.group(1).strip()
        elif 'Historical Baseline:' in line:
            match = re.search(r'Historical Baseline:\s*([\d.]+)', line)
            if match:
                result['trend_analysis']['historical_baseline'] = float(match.group(1))
        elif 'Forecast Average:' in line:
            match = re.search(r'Forecast Average:\s*([\d.]+)', line)
            if match:
                result['trend_analysis']['forecast_average'] = float(match.group(1))
        elif 'Growth Rate:' in line:
            match = re.search(r'Growth Rate:\s*([+-]?[\d.]+)%', line)
            if match:
                result['trend_analysis']['growth_rate'] = float(match.group(1))
    
    # Extract recommendation
    for line in lines:
        if 'Status:' in line and 'RECOMMENDATION' not in line:
            match = re.search(r'Status:\s*(.+)', line)
            if match:
                result['recommendation']['status'] = match.group(1).strip()
        elif 'Explanation:' in line:
            match = re.search(r'Explanation:\s*(.+)', line)
            if match:
                result['recommendation']['explanation'] = match.group(1).strip()
    
    return result

@app.route('/api/status')
def get_status():
    """Get system status"""
    global system_initialized
    
    return jsonify({
        'status': 'success',
        'initialized': system_initialized
    })

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)