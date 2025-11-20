"""
Topic Modeling Module for NASA Patent Forecasting
This module handles topic discovery using TF-IDF vectorization and Non-negative Matrix Factorization (NMF).
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import re
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def discover_topics(dataframe, num_topics=15):
    """
    Discover topics from patent text data using TF-IDF and NMF.
    
    This function implements a complete topic modeling pipeline:
    1. Text preprocessing and cleaning
    2. TF-IDF vectorization to convert text to numerical features
    3. NMF (Non-negative Matrix Factorization) to discover topics
    4. Topic assignment to each patent
    5. Keyword extraction for each topic
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing 'text' column with patent descriptions
        num_topics (int): Number of topics to discover (default: 15)
        
    Returns:
        tuple: (updated_dataframe, vectorizer, nmf_model, topic_keywords)
            - updated_dataframe: Original DataFrame with added 'topic' column
            - vectorizer: Trained TfidfVectorizer object
            - nmf_model: Trained NMF model
            - topic_keywords: Dictionary mapping topic IDs to top keywords
    """
    
    print(f"Starting topic discovery with {num_topics} topics...")
    print(f"Processing {len(dataframe)} patent documents...")
    
    # Step 1: Prepare text data
    texts = dataframe['text'].tolist()
    
    # Step 2: Text preprocessing function
    def preprocess_text(text):
        """
        Clean and preprocess text for better topic modeling.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Cleaned and preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, keep only letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short words (less than 2 characters)
        words = text.split()
        words = [word for word in words if len(word) >= 2]
        text = ' '.join(words)
        
        return text
    
    # Apply preprocessing to all texts
    print("Preprocessing text data...")
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Step 3: TF-IDF Vectorization
    print("Creating TF-IDF vectors...")
    
    # Define stopwords (common words that don't help distinguish topics)
    # These are words that appear frequently but don't carry much meaning
    custom_stopwords = [
        'system', 'method', 'apparatus', 'device', 'invention', 'present',
        'said', 'comprising', 'include', 'includes', 'including', 'patent',
        'application', 'embodiment', 'invention', 'disclosure', 'description',
        'figure', 'example', 'preferred', 'according', 'provided', 'configured',
        'adapted', 'arranged', 'disposed', 'wherein', 'therefore', 'however',
        'furthermore', 'moreover', 'additionally', 'respectively', 'particularly',
        'specifically', 'generally', 'substantially', 'approximately', 'about'
    ]
    
    # Initialize TF-IDF Vectorizer with optimized parameters
    vectorizer = TfidfVectorizer(
        max_features=5000,          # Limit to top 5000 most important words
        min_df=2,                   # Word must appear in at least 2 documents
        max_df=0.95,                # Ignore words appearing in >95% of documents
        stop_words='english',       # Remove common English stopwords
        ngram_range=(1, 2),         # Use both single words and two-word phrases
        lowercase=True,             # Convert to lowercase
        strip_accents='unicode'     # Remove accents from characters
    )
    
    # Get English stopwords and add custom ones
    english_stopwords = set(vectorizer.get_stop_words()) if hasattr(vectorizer, 'get_stop_words') else set()
    all_stopwords = list(english_stopwords.union(set(custom_stopwords)))
    
    # Re-initialize vectorizer with custom stopwords
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        stop_words=all_stopwords,   # Use combined stopwords as list
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents='unicode'
    )
    
    # Transform text to TF-IDF matrix
    # This creates a matrix where each row is a document and each column is a word/phrase
    # Values represent the importance of each word in each document
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Step 4: Apply NMF for topic discovery
    print("Applying Non-negative Matrix Factorization (NMF)...")
    
    # Initialize NMF model
    # NMF decomposes the TF-IDF matrix into topics and their word distributions
    nmf_model = NMF(
        n_components=num_topics,    # Number of topics to discover
        random_state=42,            # For reproducible results
        init='nndsvd',              # Initialization method for better convergence
        max_iter=500                # Maximum iterations for convergence
    )
    
    # Fit the model and get topic representations for each document
    doc_topic_matrix = nmf_model.fit_transform(tfidf_matrix)
    
    # Step 5: Assign topics to documents
    print("Assigning topics to documents...")
    
    # For each document, find the topic with the highest probability
    topic_assignments = np.argmax(doc_topic_matrix, axis=1)
    
    # Add topic assignments to the dataframe
    dataframe_copy = dataframe.copy()
    dataframe_copy['topic'] = topic_assignments
    
    # Step 6: Extract top keywords for each topic
    print("Extracting keywords for each topic...")
    
    # Get feature names (words/phrases) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the topic-word matrix from NMF
    topic_word_matrix = nmf_model.components_
    
    # Dictionary to store top keywords for each topic
    topic_keywords = {}
    
    # For each topic, find the top 10 most important keywords
    for topic_idx in range(num_topics):
        # Get word importance scores for this topic
        topic_words = topic_word_matrix[topic_idx]
        
        # Find indices of top 10 words
        top_word_indices = topic_words.argsort()[-10:][::-1]
        
        # Get the actual words and their scores
        top_keywords = []
        for word_idx in top_word_indices:
            word = feature_names[word_idx]
            score = topic_words[word_idx]
            top_keywords.append((word, score))
        
        # Store keywords for this topic
        topic_keywords[topic_idx] = top_keywords
    
    # Step 7: Display topic summary
    print(f"\nTopic Discovery Results:")
    print(f"Successfully discovered {num_topics} topics")
    
    # Show topic distribution
    topic_counts = dataframe_copy['topic'].value_counts().sort_index()
    print(f"\nTopic distribution:")
    for topic_id, count in topic_counts.items():
        percentage = (count / len(dataframe_copy)) * 100
        keywords = [word for word, score in topic_keywords[topic_id][:5]]
        print(f"Topic {topic_id}: {count} patents ({percentage:.1f}%) - Keywords: {', '.join(keywords)}")
    
    # Step 8: Calculate topic coherence metrics
    print(f"\nTopic modeling quality metrics:")
    
    # Calculate average topic probability (higher is better)
    max_probs = np.max(doc_topic_matrix, axis=1)
    avg_max_prob = np.mean(max_probs)
    print(f"Average maximum topic probability: {avg_max_prob:.3f}")
    
    # Calculate topic diversity (should be balanced)
    topic_diversity = len(np.unique(topic_assignments)) / num_topics
    print(f"Topic diversity (topics actually used): {topic_diversity:.3f}")
    
    return dataframe_copy, vectorizer, nmf_model, topic_keywords


def display_topic_details(topic_keywords, num_topics_to_show=None):
    """
    Display detailed information about discovered topics.
    
    Args:
        topic_keywords (dict): Dictionary mapping topic IDs to keywords
        num_topics_to_show (int): Number of topics to display (default: all)
    """
    
    if num_topics_to_show is None:
        num_topics_to_show = len(topic_keywords)
    
    print(f"\nDetailed Topic Analysis:")
    print("=" * 80)
    
    for topic_id in range(min(num_topics_to_show, len(topic_keywords))):
        print(f"\nTopic {topic_id}:")
        print("-" * 40)
        
        keywords = topic_keywords[topic_id]
        for i, (word, score) in enumerate(keywords):
            print(f"  {i+1:2d}. {word:<20} (importance: {score:.4f})")


def predict_topic_for_text(text, vectorizer, nmf_model):
    """
    Predict the most likely topic for a new text.
    
    Args:
        text (str): New text to classify
        vectorizer: Trained TfidfVectorizer
        nmf_model: Trained NMF model
        
    Returns:
        tuple: (predicted_topic_id, topic_probabilities)
    """
    
    # Preprocess the text
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        words = [word for word in words if len(word) >= 2]
        return ' '.join(words)
    
    # Apply preprocessing
    processed_text = preprocess_text(text)
    
    # Transform to TF-IDF
    tfidf_vector = vectorizer.transform([processed_text])
    
    # Get topic probabilities
    topic_probs = nmf_model.transform(tfidf_vector)
    
    # Find the most likely topic
    predicted_topic = np.argmax(topic_probs[0])
    
    return predicted_topic, topic_probs[0]


if __name__ == "__main__":
    # Test the module independently
    print("Testing topic modeling module...")
    
    # Create sample data for testing
    sample_data = {
        'text': [
            "Solar panel energy conversion system for spacecraft power generation",
            "Advanced propulsion rocket engine design for space exploration missions",
            "Satellite communication antenna array technology for deep space",
            "Life support environmental control system for crew safety",
            "Navigation guidance computer software for autonomous flight control"
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    try:
        # Test topic discovery
        df_with_topics, vectorizer, nmf_model, keywords = discover_topics(df, num_topics=3)
        
        # Display results
        display_topic_details(keywords)
        
        # Test prediction for new text
        new_text = "innovative battery technology for energy storage"
        topic_id, probs = predict_topic_for_text(new_text, vectorizer, nmf_model)
        print(f"\nNew text predicted topic: {topic_id}")
        
        print("Topic modeling test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")