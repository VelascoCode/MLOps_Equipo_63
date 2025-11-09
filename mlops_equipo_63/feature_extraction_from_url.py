import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import re
import random
from datetime import datetime, timedelta

def extract_features(url):
    """
    Extracts a complete set of NLP-based features from a news article URL.
    Uses random placeholders for features that cannot be derived directly.

    Args:
        url (str): The URL of the news article.

    Returns:
        dict: A dictionary of extracted features.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Basic Text Extraction ---
        title = soup.find('title').get_text() if soup.find('title') else ''
        content = ' '.join([p.get_text() for p in soup.find_all('p')])

        # --- Tokenization ---
        title_tokens = word_tokenize(title)
        content_tokens = word_tokenize(content)
        
        stop_words = set(stopwords.words('english'))

        # --- Feature Calculation ---
        features = {}

        # n_tokens_title
        features['n_tokens_title'] = len(title_tokens)

        # n_tokens_content
        features['n_tokens_content'] = len(content_tokens)

        # n_unique_tokens
        features['n_unique_tokens'] = len(set(content_tokens)) / features['n_tokens_content'] if features['n_tokens_content'] > 0 else 0
            
        # n_non_stop_words
        non_stop_words = [word for word in content_tokens if word.lower() not in stop_words and word.isalpha()]
        features['n_non_stop_words'] = len(non_stop_words) / features['n_tokens_content'] if features['n_tokens_content'] > 0 else 0

        # n_non_stop_unique_tokens
        features['n_non_stop_unique_tokens'] = len(set(non_stop_words)) / len(non_stop_words) if len(non_stop_words) > 0 else 0

        # num_hrefs, num_imgs, num_videos
        features['num_hrefs'] = len(soup.find_all('a'))
        features['num_imgs'] = len(soup.find_all('img'))
        features['num_videos'] = len(soup.find_all('video'))

        # num_self_hrefs (a simple approximation)
        domain = re.findall(r'https?://([^/]+)', url)
        features['num_self_hrefs'] = len([a for a in soup.find_all('a', href=True) if domain and domain[0] in a['href']])

        # average_token_length
        features['average_token_length'] = sum(len(word) for word in content_tokens) / features['n_tokens_content'] if features['n_tokens_content'] > 0 else 0
            
        # num_keywords (approximated by counting nouns and proper nouns in the title)
        tagged_title = nltk.pos_tag(title_tokens)
        keywords = [word for word, tag in tagged_title if tag.startswith('NN')]
        features['num_keywords'] = len(keywords)
        
        # --- Data Channel (simple keyword-based classification) ---
        text_lower = (title + ' ' + content).lower()
        channels = {
            'data_channel_is_lifestyle': ['fashion', 'style', 'travel', 'food', 'home'],
            'data_channel_is_entertainment': ['movie', 'music', 'celebrity', 'tv', 'show'],
            'data_channel_is_bus': ['business', 'finance', 'market', 'stocks', 'economy'],
            'data_channel_is_socmed': ['facebook', 'twitter', 'instagram', 'social media'],
            'data_channel_is_tech': ['technology', 'apple', 'google', 'microsoft', 'tech'],
            'data_channel_is_world': ['world', 'politics', 'global', 'international']
        }
        for channel, keywords in channels.items():
            features[channel] = 1 if any(keyword in text_lower for keyword in keywords) else 0

        # --- Placeholder Keyword Statistics ---
        # These require statistics from the original dataset, so we use random values.
        features['kw_min_min'] = random.randint(-1, 200)
        features['kw_max_min'] = random.uniform(0, 10000)
        features['kw_avg_min'] = random.uniform(0, 5000)
        features['kw_min_max'] = random.uniform(0, 100000)
        features['kw_max_max'] = random.randint(100000, 850000)
        features['kw_avg_max'] = random.uniform(10000, 500000)
        features['kw_min_avg'] = random.uniform(0, 5000)
        features['kw_max_avg'] = random.uniform(2000, 20000)
        features['kw_avg_avg'] = random.uniform(1000, 10000)
        
        # --- Placeholder Self-Reference Share Statistics ---
        # These also require statistics from the original dataset, so we use random values.
        features['self_reference_min_shares'] = random.uniform(0, 80000)
        features['self_reference_max_shares'] = random.uniform(100, 850000)
        features['self_reference_avg_sharess'] = random.uniform(100, 200000)

        # --- Weekday Features (from random date) ---
        # Since parsing the real date is unreliable, we generate a random date in the last year.
        random_days_ago = random.randint(0, 365)
        article_date = datetime.now() - timedelta(days=random_days_ago)
        weekday = article_date.weekday() # Monday is 0 and Sunday is 6
        
        features['weekday_is_monday'] = 1 if weekday == 0 else 0
        features['weekday_is_tuesday'] = 1 if weekday == 1 else 0
        features['weekday_is_wednesday'] = 1 if weekday == 2 else 0
        features['weekday_is_thursday'] = 1 if weekday == 3 else 0
        features['weekday_is_friday'] = 1 if weekday == 4 else 0
        features['weekday_is_saturday'] = 1 if weekday == 5 else 0
        features['weekday_is_sunday'] = 1 if weekday == 6 else 0
        features['is_weekend'] = 1 if weekday >= 5 else 0

        # --- Placeholder LDA Topic Modeling Features ---
        # Since we don't have the pre-trained LDA model, we use random topic distributions.
        features['LDA_00'] = random.random()
        features['LDA_01'] = random.random()
        features['LDA_02'] = random.random()
        features['LDA_03'] = random.random()
        features['LDA_04'] = random.random()

        # --- Sentiment Analysis ---
        title_blob = TextBlob(title)
        content_blob = TextBlob(content)

        features['global_subjectivity'] = content_blob.sentiment.subjectivity
        features['global_sentiment_polarity'] = content_blob.sentiment.polarity

        positive_words = [word for word in content_blob.words if TextBlob(word).sentiment.polarity > 0]
        negative_words = [word for word in content_blob.words if TextBlob(word).sentiment.polarity < 0]
        
        features['global_rate_positive_words'] = len(positive_words) / features['n_tokens_content'] if features['n_tokens_content'] > 0 else 0
        features['global_rate_negative_words'] = len(negative_words) / features['n_tokens_content'] if features['n_tokens_content'] > 0 else 0
        
        non_stop_word_count = len(non_stop_words)
        features['rate_positive_words'] = len(positive_words) / non_stop_word_count if non_stop_word_count > 0 else 0
        features['rate_negative_words'] = len(negative_words) / non_stop_word_count if non_stop_word_count > 0 else 0

        features['avg_positive_polarity'] = sum(TextBlob(w).sentiment.polarity for w in positive_words) / len(positive_words) if positive_words else 0
        features['min_positive_polarity'] = min(TextBlob(w).sentiment.polarity for w in positive_words) if positive_words else 0
        features['max_positive_polarity'] = max(TextBlob(w).sentiment.polarity for w in positive_words) if positive_words else 0

        features['avg_negative_polarity'] = sum(TextBlob(w).sentiment.polarity for w in negative_words) / len(negative_words) if negative_words else 0
        features['min_negative_polarity'] = min(TextBlob(w).sentiment.polarity for w in negative_words) if negative_words else 0
        features['max_negative_polarity'] = max(TextBlob(w).sentiment.polarity for w in negative_words) if negative_words else 0
        
        features['title_subjectivity'] = title_blob.sentiment.subjectivity
        features['title_sentiment_polarity'] = title_blob.sentiment.polarity
        features['abs_title_subjectivity'] = abs(features['title_subjectivity'] - 0.5)
        features['abs_title_sentiment_polarity'] = abs(features['title_sentiment_polarity'])
            
        return features

    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")
        return None

# --- Example Usage ---
if __name__ == '__main__':
    # You will need to install the required libraries first:
    # pip install requests beautifulsoup4 nltk textblob
    # And download NLTK data:
    # import nltk
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('averaged_perceptron_tagger')

    article_url = "https://techcrunch.com/2024/02/21/google-releases-gemma-its-new-generation-of-open-models/"
    
    extracted_features = extract_features(article_url)
    
    if extracted_features:
        # Print all 58 features to see the complete output
        for feature, value in extracted_features.items():
            print(f"{feature}: {value}")