"""
Trend Evaluation Module for NASA Patent Forecasting
This module evaluates new patent ideas against predicted technology trends.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any


def evaluate_new_idea(idea_text, vectorizer, nmf_model, forecasts, last_historical_year):
    """
    Evaluate whether a new patent idea aligns with high-growth technology trends.
    
    This function implements the complete evaluation pipeline:
    1. Uses trained NLP models to determine which topic the new idea belongs to
    2. Retrieves the forecast for that topic
    3. Analyzes the growth trend of the forecast
    4. Provides a clear recommendation about the idea's alignment with trends
    
    Args:
        idea_text (str): The new patent idea text to evaluate
        vectorizer: Trained TfidfVectorizer from topic modeling
        nmf_model: Trained NMF model from topic modeling
        forecasts (dict): Dictionary containing forecasts for all topics
        last_historical_year (int): The last year of historical data
        
    Returns:
        str: Detailed evaluation result with recommendation
    """
    
    print(f"Evaluating new patent idea...")
    print(f"Idea text: '{idea_text[:100]}{'...' if len(idea_text) > 100 else ''}'")
    
    # Step 1: Predict which topic the new idea belongs to
    predicted_topic, topic_probabilities = predict_topic_for_new_idea(
        idea_text, vectorizer, nmf_model
    )
    
    # Step 2: Get the forecast for the predicted topic
    topic_name = f"Topic_{predicted_topic}"
    
    if topic_name not in forecasts:
        return f"Error: No forecast available for {topic_name}. Cannot evaluate trend."
    
    forecast_data = forecasts[topic_name]
    
    # Step 3: Analyze the trend
    trend_analysis = analyze_forecast_trend(
        forecast_data, last_historical_year, topic_name
    )
    
    # Step 4: Determine confidence level based on topic assignment probability
    max_probability = np.max(topic_probabilities)
    confidence_level = get_confidence_level(max_probability)
    
    # Step 5: Generate comprehensive evaluation report
    evaluation_report = generate_evaluation_report(
        idea_text, predicted_topic, topic_probabilities, 
        trend_analysis, confidence_level, forecast_data
    )
    
    return evaluation_report


def predict_topic_for_new_idea(idea_text, vectorizer, nmf_model):
    """
    Predict which topic a new patent idea belongs to using trained NLP models.
    
    Args:
        idea_text (str): Text of the new patent idea
        vectorizer: Trained TfidfVectorizer
        nmf_model: Trained NMF model
        
    Returns:
        tuple: (predicted_topic_id, topic_probabilities_array)
    """
    
    # Import the preprocessing function from topic_modeler
    import re
    
    def preprocess_text(text):
        """Clean and preprocess text for topic prediction."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        words = [word for word in words if len(word) >= 2]
        return ' '.join(words)
    
    # Preprocess the input text
    processed_text = preprocess_text(idea_text)
    
    if not processed_text.strip():
        raise ValueError("Input text is empty after preprocessing")
    
    # Transform text to TF-IDF vector
    tfidf_vector = vectorizer.transform([processed_text])
    
    # Get topic probabilities from NMF
    topic_probabilities = nmf_model.transform(tfidf_vector)[0]
    
    # Find the most likely topic
    predicted_topic = np.argmax(topic_probabilities)
    
    print(f"Topic prediction results:")
    print(f"  Most likely topic: Topic_{predicted_topic}")
    print(f"  Confidence: {topic_probabilities[predicted_topic]:.3f}")
    
    # Show top 3 topic probabilities
    top_topics = np.argsort(topic_probabilities)[-3:][::-1]
    print(f"  Top 3 topic matches:")
    for i, topic_id in enumerate(top_topics):
        print(f"    {i+1}. Topic_{topic_id}: {topic_probabilities[topic_id]:.3f}")
    
    return predicted_topic, topic_probabilities


def analyze_forecast_trend(forecast_data, last_historical_year, topic_name):
    """
    Analyze the growth trend of a topic's forecast.
    
    Args:
        forecast_data (dict): Forecast data for the topic
        last_historical_year (int): Last year of historical data
        topic_name (str): Name of the topic
        
    Returns:
        dict: Analysis results including trend classification and metrics
    """
    
    print(f"\nAnalyzing trend for {topic_name}...")
    
    historical_data = forecast_data['historical_data']
    forecasts = forecast_data['forecasts']
    
    # Get recent historical baseline (last 3 years average)
    recent_years = min(3, len(historical_data))
    recent_historical_avg = np.mean(historical_data[-recent_years:])
    
    # Get last historical year value
    last_historical_value = historical_data[-1]
    
    # Calculate forecast metrics
    forecast_avg = np.mean(forecasts)
    forecast_max = np.max(forecasts)
    forecast_min = np.min(forecasts)
    
    # Calculate growth metrics
    
    # 1. Historical to forecast growth
    historical_to_forecast_growth = ((forecast_avg - last_historical_value) / last_historical_value) * 100 if last_historical_value > 0 else 0
    
    # 2. Recent historical average to forecast average growth
    recent_to_forecast_growth = ((forecast_avg - recent_historical_avg) / recent_historical_avg) * 100 if recent_historical_avg > 0 else 0
    
    # 3. Forecast internal trend (first year vs last year of forecast)
    forecast_internal_growth = ((forecasts[-1] - forecasts[0]) / forecasts[0]) * 100 if forecasts[0] > 0 else 0
    
    # 4. Calculate trend momentum
    if len(historical_data) >= 5:
        # Compare last 3 years vs previous 3 years
        early_period = np.mean(historical_data[-6:-3]) if len(historical_data) >= 6 else np.mean(historical_data[:-3])
        late_period = np.mean(historical_data[-3:])
        historical_momentum = ((late_period - early_period) / early_period) * 100 if early_period > 0 else 0
    else:
        historical_momentum = 0
    
    # Classify the trend
    trend_classification = classify_trend(
        historical_to_forecast_growth,
        recent_to_forecast_growth,
        forecast_internal_growth,
        historical_momentum
    )
    
    # Compile analysis results
    analysis = {
        'topic_name': topic_name,
        'last_historical_value': last_historical_value,
        'recent_historical_avg': recent_historical_avg,
        'forecast_avg': forecast_avg,
        'forecast_range': (forecast_min, forecast_max),
        'historical_to_forecast_growth': historical_to_forecast_growth,
        'recent_to_forecast_growth': recent_to_forecast_growth,
        'forecast_internal_growth': forecast_internal_growth,
        'historical_momentum': historical_momentum,
        'trend_classification': trend_classification,
        'forecasts': forecasts,
        'forecast_years': forecast_data['forecast_years']
    }
    
    print(f"Trend analysis results:")
    print(f"  Last historical value: {last_historical_value:.1f}")
    print(f"  Average forecast: {forecast_avg:.1f}")
    print(f"  Growth rate: {historical_to_forecast_growth:.1f}%")
    print(f"  Trend classification: {trend_classification}")
    
    return analysis


def classify_trend(hist_to_forecast_growth, recent_to_forecast_growth, 
                  forecast_internal_growth, historical_momentum):
    """
    Classify the trend based on multiple growth metrics.
    
    Args:
        hist_to_forecast_growth (float): Historical to forecast growth percentage
        recent_to_forecast_growth (float): Recent historical to forecast growth percentage
        forecast_internal_growth (float): Growth within the forecast period
        historical_momentum (float): Historical momentum percentage
        
    Returns:
        str: Trend classification
    """
    
    # Define thresholds for classification
    HIGH_GROWTH_THRESHOLD = 20  # >20% growth
    MODERATE_GROWTH_THRESHOLD = 10  # 10-20% growth
    LOW_GROWTH_THRESHOLD = 5   # 5-10% growth
    STABLE_THRESHOLD = -5      # -5% to 5% is stable
    
    # Weight the different growth metrics
    combined_growth = (
        hist_to_forecast_growth * 0.4 +
        recent_to_forecast_growth * 0.3 +
        forecast_internal_growth * 0.2 +
        historical_momentum * 0.1
    )
    
    # Classify based on combined growth
    if combined_growth > HIGH_GROWTH_THRESHOLD:
        return "HIGH-GROWTH (Booming)"
    elif combined_growth > MODERATE_GROWTH_THRESHOLD:
        return "MODERATE-GROWTH (Growing)"
    elif combined_growth > LOW_GROWTH_THRESHOLD:
        return "LOW-GROWTH (Slow Growth)"
    elif combined_growth > STABLE_THRESHOLD:
        return "STABLE (Steady)"
    else:
        return "DECLINING (Decreasing)"


def get_confidence_level(max_probability):
    """
    Determine confidence level based on topic assignment probability.
    
    Args:
        max_probability (float): Maximum topic probability
        
    Returns:
        str: Confidence level description
    """
    
    if max_probability > 0.6:
        return "HIGH"
    elif max_probability > 0.4:
        return "MEDIUM"
    elif max_probability > 0.2:
        return "LOW"
    else:
        return "VERY LOW"


def generate_evaluation_report(idea_text, predicted_topic, topic_probabilities, 
                             trend_analysis, confidence_level, forecast_data):
    """
    Generate a comprehensive evaluation report for the new patent idea.
    
    Args:
        idea_text (str): Original idea text
        predicted_topic (int): Predicted topic ID
        topic_probabilities (np.array): Topic probability scores
        trend_analysis (dict): Trend analysis results
        confidence_level (str): Confidence level of prediction
        forecast_data (dict): Complete forecast data
        
    Returns:
        str: Formatted evaluation report
    """
    
    report = []
    report.append("="*80)
    report.append("PATENT IDEA TREND EVALUATION REPORT")
    report.append("="*80)
    
    # Section 1: Idea Summary
    report.append(f"\n📝 PATENT IDEA SUMMARY:")
    report.append(f"Text: {idea_text}")
    report.append(f"Length: {len(idea_text)} characters")
    
    # Section 2: Topic Classification
    report.append(f"\n🎯 TOPIC CLASSIFICATION:")
    report.append(f"Predicted Topic: Topic_{predicted_topic}")
    report.append(f"Classification Confidence: {confidence_level}")
    report.append(f"Topic Probability: {topic_probabilities[predicted_topic]:.3f}")
    
    # Show alternative topics if confidence is not high
    if confidence_level in ["LOW", "VERY LOW"]:
        top_topics = np.argsort(topic_probabilities)[-3:][::-1]
        report.append(f"\nAlternative Topic Matches:")
        for i, topic_id in enumerate(top_topics[:3]):
            report.append(f"  {i+1}. Topic_{topic_id}: {topic_probabilities[topic_id]:.3f}")
    
    # Section 3: Trend Analysis
    report.append(f"\n📈 TREND ANALYSIS:")
    report.append(f"Trend Classification: {trend_analysis['trend_classification']}")
    report.append(f"Historical Baseline: {trend_analysis['last_historical_value']:.1f} patents")
    report.append(f"Forecast Average: {trend_analysis['forecast_avg']:.1f} patents")
    report.append(f"Growth Rate: {trend_analysis['historical_to_forecast_growth']:.1f}%")
    
    # Section 4: Detailed Forecast
    report.append(f"\n🔮 FORECAST DETAILS:")
    forecast_years = trend_analysis['forecast_years']
    forecasts = trend_analysis['forecasts']
    
    for year, forecast in zip(forecast_years, forecasts):
        report.append(f"  {year}: {forecast:.1f} patents")
    
    report.append(f"Forecast Range: {trend_analysis['forecast_range'][0]:.1f} - {trend_analysis['forecast_range'][1]:.1f} patents")
    
    # Section 5: Recommendation
    report.append(f"\n💡 RECOMMENDATION:")
    
    trend_class = trend_analysis['trend_classification']
    growth_rate = trend_analysis['historical_to_forecast_growth']
    
    if "HIGH-GROWTH" in trend_class or "BOOMING" in trend_class:
        recommendation = "🚀 HIGHLY RECOMMENDED"
        explanation = "This idea aligns with a rapidly growing technology trend. Excellent opportunity for innovation and market success."
    elif "MODERATE-GROWTH" in trend_class or "GROWING" in trend_class:
        recommendation = "✅ RECOMMENDED"
        explanation = "This idea aligns with a growing technology trend. Good potential for development and market adoption."
    elif "LOW-GROWTH" in trend_class:
        recommendation = "⚠️  CAUTIOUSLY RECOMMENDED"
        explanation = "This idea aligns with a slowly growing trend. Consider market timing and differentiation strategies."
    elif "STABLE" in trend_class:
        recommendation = "🔄 NEUTRAL"
        explanation = "This idea aligns with a stable technology area. Success will depend on innovation quality and execution."
    else:  # DECLINING
        recommendation = "❌ NOT RECOMMENDED"
        explanation = "This idea aligns with a declining technology trend. Consider pivoting to more promising areas."
    
    report.append(f"Status: {recommendation}")
    report.append(f"Explanation: {explanation}")
    
    # Section 6: Risk Assessment
    report.append(f"\n⚠️  RISK ASSESSMENT:")
    
    risks = []
    
    if confidence_level in ["LOW", "VERY LOW"]:
        risks.append("• Low topic classification confidence - idea may not fit well into existing categories")
    
    if abs(growth_rate) > 50:
        risks.append("• High volatility in forecast - predictions may be uncertain")
    
    if trend_analysis['last_historical_value'] < 5:
        risks.append("• Low historical activity in this topic - limited market validation")
    
    forecast_variance = np.var(forecasts)
    if forecast_variance > trend_analysis['forecast_avg']:
        risks.append("• High variance in forecast - uncertain future trajectory")
    
    if not risks:
        risks.append("• No significant risks identified based on available data")
    
    for risk in risks:
        report.append(risk)
    
    # Section 7: Additional Insights
    report.append(f"\n🔍 ADDITIONAL INSIGHTS:")
    
    if trend_analysis['historical_momentum'] > 10:
        report.append("• Strong historical momentum suggests accelerating interest in this technology area")
    elif trend_analysis['historical_momentum'] < -10:
        report.append("• Historical momentum shows declining interest - trend may be maturing")
    
    if trend_analysis['forecast_internal_growth'] > 15:
        report.append("• Forecast shows accelerating growth throughout the prediction period")
    elif trend_analysis['forecast_internal_growth'] < -15:
        report.append("• Forecast shows growth may peak early in the prediction period")
    
    # Section 8: Summary
    report.append(f"\n📋 EXECUTIVE SUMMARY:")
    summary = f"The patent idea '{idea_text[:50]}...' belongs to Topic_{predicted_topic} "
    summary += f"with {confidence_level.lower()} confidence ({topic_probabilities[predicted_topic]:.3f}). "
    summary += f"This technology area is classified as {trend_class.lower()} "
    summary += f"with an expected growth rate of {growth_rate:.1f}%. "
    summary += f"Overall recommendation: {recommendation.split(' ', 1)[1]}."
    
    report.append(summary)
    
    report.append("="*80)
    
    return "\n".join(report)


def batch_evaluate_ideas(ideas_list, vectorizer, nmf_model, forecasts, last_historical_year):
    """
    Evaluate multiple patent ideas at once.
    
    Args:
        ideas_list (list): List of patent idea texts
        vectorizer: Trained TfidfVectorizer
        nmf_model: Trained NMF model
        forecasts (dict): Forecasts for all topics
        last_historical_year (int): Last year of historical data
        
    Returns:
        list: List of evaluation results
    """
    
    print(f"Batch evaluating {len(ideas_list)} patent ideas...")
    
    results = []
    
    for i, idea in enumerate(ideas_list):
        print(f"\nEvaluating idea {i+1}/{len(ideas_list)}...")
        
        try:
            result = evaluate_new_idea(idea, vectorizer, nmf_model, forecasts, last_historical_year)
            results.append({
                'idea_text': idea,
                'evaluation': result,
                'status': 'success'
            })
        except Exception as e:
            print(f"Error evaluating idea {i+1}: {e}")
            results.append({
                'idea_text': idea,
                'evaluation': f"Error: {str(e)}",
                'status': 'error'
            })
    
    return results


if __name__ == "__main__":
    # Test the trend evaluation module
    print("Testing trend evaluation module...")
    
    # This would normally be run with real trained models and forecasts
    # For testing, we'll create mock data
    
    test_idea = "Advanced solar panel technology for spacecraft energy systems"
    
    print(f"Test completed - trend evaluation module is ready!")
    print(f"In actual use, this would evaluate: '{test_idea}'")