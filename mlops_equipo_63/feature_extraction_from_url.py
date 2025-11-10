"""Feature extraction helpers for the prediction API.

This module provides:
- extract_features(url, fill_random=False): returns a dict of features extracted from an article URL.
- ensure_feature_vector(features, feature_names): aligns the returned dict to a list of feature names (fills missing keys with zeros).
- extract_and_predict(url, model, feature_names=None, fill_random=False): convenience helper to extract features and call a loaded model.

Notes:
- By default dataset-specific aggregated statistics (kw_*, self_reference_*, LDA_*) are filled with 0.0 for determinism.
  Set fill_random=True to keep the historical behavior of using random placeholders (not recommended for production).
"""

from typing import Dict, List, Optional, Any
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import numpy as np
import logging

try:
    # Optional heavy deps. We import lazily where used to avoid import-time failures in environments
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from textblob import TextBlob
except Exception:
    # We'll raise later if functions requiring these libs are invoked
    nltk = None
    stopwords = None
    word_tokenize = None
    TextBlob = None

_logger = logging.getLogger(__name__)


def _safe_nltk_setup():
    """Ensure the minimal NLTK data is available; downloads if missing.

    This may attempt to download resources if not already present.
    """
    if nltk is None:
        raise RuntimeError("nltk or textblob not installed; install nltk and textblob to use NLP features")
    try:
        # these calls will raise LookupError if data is missing
        word_tokenize("test")
        stopwords.words('english')
        nltk.pos_tag(["test"])  # test tagger
    except LookupError:
        _logger.info("Downloading missing NLTK data (punkt, stopwords, averaged_perceptron_tagger)")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')


def extract_features(url: str, fill_random: bool = False) -> Optional[Dict[str, Any]]:
    """Extract a set of features from the article at `url`.

    Parameters
    ----------
    url: str
        Article URL to fetch and parse.
    fill_random: bool
        If True, fill dataset-specific placeholder features with random values (legacy behaviour).
        If False (default), placeholders are filled with deterministic zeros.

    Returns
    -------
    dict
        Mapping feature_name -> value. Returns None on error.
    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'html.parser')

        title = soup.title.get_text(strip=True) if soup.title else ''
        content = ' '.join(p.get_text(separator=' ', strip=True) for p in soup.find_all('p'))

        # Ensure NLP data is available
        if nltk is None or TextBlob is None:
            raise RuntimeError("Missing optional dependencies: install nltk and textblob to enable NLP features")

        _safe_nltk_setup()

        title_tokens = word_tokenize(title) if title else []
        content_tokens = word_tokenize(content) if content else []
        stop_words = set(stopwords.words('english'))

        features: Dict[str, Any] = {}

        # Basic counts
        features['n_tokens_title'] = len(title_tokens)
        features['n_tokens_content'] = len(content_tokens)
        features['n_unique_tokens'] = (len(set(content_tokens)) / len(content_tokens)) if content_tokens else 0.0

        non_stop_words = [w for w in content_tokens if w.isalpha() and w.lower() not in stop_words]
        features['n_non_stop_words'] = (len(non_stop_words) / len(content_tokens)) if content_tokens else 0.0
        features['n_non_stop_unique_tokens'] = (len(set(non_stop_words)) / len(non_stop_words)) if non_stop_words else 0.0

        # counts of elements
        features['num_hrefs'] = len(soup.find_all('a'))
        features['num_imgs'] = len(soup.find_all('img'))
        features['num_videos'] = len(soup.find_all('video'))

        # self hrefs
        parsed = urlparse(url)
        domain = parsed.netloc
        links = [a.get('href', '') for a in soup.find_all('a', href=True)]
        features['num_self_hrefs'] = sum(1 for href in links if domain and domain in href)

        # average token length
        features['average_token_length'] = (sum(len(w) for w in content_tokens) / len(content_tokens)) if content_tokens else 0.0

        # keywords approx: nouns in title
        try:
            tagged = nltk.pos_tag(title_tokens) if title_tokens else []
            keywords = [w for w, t in tagged if t.startswith('NN')]
            features['num_keywords'] = len(keywords)
        except Exception:
            features['num_keywords'] = 0

        # simple channel flags
        text_lower = (title + ' ' + content).lower()
        channels = {
            'data_channel_is_lifestyle': ['fashion', 'style', 'travel', 'food', 'home'],
            'data_channel_is_entertainment': ['movie', 'music', 'celebrity', 'tv', 'show'],
            'data_channel_is_bus': ['business', 'finance', 'market', 'stocks', 'economy'],
            'data_channel_is_socmed': ['facebook', 'twitter', 'instagram', 'social media'],
            'data_channel_is_tech': ['technology', 'apple', 'google', 'microsoft', 'tech'],
            'data_channel_is_world': ['world', 'politics', 'global', 'international']
        }
        for ch, kws in channels.items():
            features[ch] = 1 if any(k in text_lower for k in kws) else 0

        # Placeholder aggregated stats: deterministic zeros by default (safer for production)
        if fill_random:
            import random
            features.update({
                'kw_min_min': float(random.randint(-1, 200)),
                'kw_max_min': float(random.uniform(0, 10000)),
                'kw_avg_min': float(random.uniform(0, 5000)),
                'kw_min_max': float(random.uniform(0, 100000)),
                'kw_max_max': float(random.randint(100000, 850000)),
                'kw_avg_max': float(random.uniform(10000, 500000)),
                'kw_min_avg': float(random.uniform(0, 5000)),
                'kw_max_avg': float(random.uniform(2000, 20000)),
                'kw_avg_avg': float(random.uniform(1000, 10000)),
                'self_reference_min_shares': float(random.uniform(0, 80000)),
                'self_reference_max_shares': float(random.uniform(100, 850000)),
                'self_reference_avg_sharess': float(random.uniform(100, 200000)),
                'LDA_00': random.random(), 'LDA_01': random.random(), 'LDA_02': random.random(),
                'LDA_03': random.random(), 'LDA_04': random.random()
            })
        else:
            # deterministic defaults
            features.update({
                'kw_min_min': 0.0, 'kw_max_min': 0.0, 'kw_avg_min': 0.0,
                'kw_min_max': 0.0, 'kw_max_max': 0.0, 'kw_avg_max': 0.0,
                'kw_min_avg': 0.0, 'kw_max_avg': 0.0, 'kw_avg_avg': 0.0,
                'self_reference_min_shares': 0.0, 'self_reference_max_shares': 0.0, 'self_reference_avg_sharess': 0.0,
                'LDA_00': 0.0, 'LDA_01': 0.0, 'LDA_02': 0.0, 'LDA_03': 0.0, 'LDA_04': 0.0
            })

        # sentiment using TextBlob
        try:
            title_blob = TextBlob(title)
            content_blob = TextBlob(content)
            features['global_subjectivity'] = float(content_blob.sentiment.subjectivity)
            features['global_sentiment_polarity'] = float(content_blob.sentiment.polarity)

            pos_words = [w for w in content_blob.words if TextBlob(w).sentiment.polarity > 0]
            neg_words = [w for w in content_blob.words if TextBlob(w).sentiment.polarity < 0]

            features['global_rate_positive_words'] = (len(pos_words) / len(content_tokens)) if content_tokens else 0.0
            features['global_rate_negative_words'] = (len(neg_words) / len(content_tokens)) if content_tokens else 0.0

            non_stop_count = len(non_stop_words)
            features['rate_positive_words'] = (len(pos_words) / non_stop_count) if non_stop_count else 0.0
            features['rate_negative_words'] = (len(neg_words) / non_stop_count) if non_stop_count else 0.0

            features['avg_positive_polarity'] = float(np.mean([TextBlob(w).sentiment.polarity for w in pos_words])) if pos_words else 0.0
            features['min_positive_polarity'] = float(np.min([TextBlob(w).sentiment.polarity for w in pos_words])) if pos_words else 0.0
            features['max_positive_polarity'] = float(np.max([TextBlob(w).sentiment.polarity for w in pos_words])) if pos_words else 0.0

            features['avg_negative_polarity'] = float(np.mean([TextBlob(w).sentiment.polarity for w in neg_words])) if neg_words else 0.0
            features['min_negative_polarity'] = float(np.min([TextBlob(w).sentiment.polarity for w in neg_words])) if neg_words else 0.0
            features['max_negative_polarity'] = float(np.max([TextBlob(w).sentiment.polarity for w in neg_words])) if neg_words else 0.0

            features['title_subjectivity'] = float(title_blob.sentiment.subjectivity)
            features['title_sentiment_polarity'] = float(title_blob.sentiment.polarity)
            features['abs_title_subjectivity'] = abs(features['title_subjectivity'] - 0.5)
            features['abs_title_sentiment_polarity'] = abs(features['title_sentiment_polarity'])
        except Exception:
            # If sentiment computation fails, provide defaults
            features.update({
                'global_subjectivity': 0.0, 'global_sentiment_polarity': 0.0,
                'global_rate_positive_words': 0.0, 'global_rate_negative_words': 0.0,
                'rate_positive_words': 0.0, 'rate_negative_words': 0.0,
                'avg_positive_polarity': 0.0, 'min_positive_polarity': 0.0, 'max_positive_polarity': 0.0,
                'avg_negative_polarity': 0.0, 'min_negative_polarity': 0.0, 'max_negative_polarity': 0.0,
                'title_subjectivity': 0.0, 'title_sentiment_polarity': 0.0,
                'abs_title_subjectivity': 0.0, 'abs_title_sentiment_polarity': 0.0
            })

        # weekday features: attempt to read OG meta date; fallback to zeros
        # (original code used random date; here we set deterministic defaults)
        features.update({
            'weekday_is_monday': 0, 'weekday_is_tuesday': 0, 'weekday_is_wednesday': 0,
            'weekday_is_thursday': 0, 'weekday_is_friday': 0, 'weekday_is_saturday': 0,
            'weekday_is_sunday': 0, 'is_weekend': 0
        })

        return features

    except Exception as e:
        _logger.exception("Failed to extract features from %s: %s", url, e)
        return None


def ensure_feature_vector(features: Dict[str, Any], feature_names: List[str], fill_value: float = 0.0) -> Dict[str, Any]:
    """Return a new dict containing keys in `feature_names` in that order.

    Missing keys are filled with `fill_value`.
    """
    out: Dict[str, Any] = {}
    for name in feature_names:
        out[name] = features.get(name, fill_value)
    return out


def extract_and_predict(url: str, model: Any, feature_names: Optional[List[str]] = None, fill_random: bool = False) -> Optional[Dict[str, Any]]:
    """Extract features from `url`, align them to `feature_names` if given, and call `model`.

    Returns a dict with keys: prediction, probability/probabilities (when available), and features
    """
    feats = extract_features(url, fill_random=fill_random)
    if feats is None:
        return None

    import pandas as pd

    if feature_names:
        vector = ensure_feature_vector(feats, feature_names, fill_value=0.0)
        df = pd.DataFrame([vector])
    else:
        df = pd.DataFrame([feats])

    # Align numeric types
    df = df.apply(pd.to_numeric, errors='ignore')

    try:
        pred = model.predict(df)
    except Exception as e:
        _logger.exception("Model prediction failed: %s", e)
        return None

    result: Dict[str, Any] = {"prediction": pred[0] if hasattr(pred, '__len__') else pred}

    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)
            if proba.shape[1] == 2:
                result['probability'] = float(proba[0, 1])
            else:
                result['probabilities'] = proba[0].tolist()
    except Exception:
        # ignore probability errors
        pass

    result['features'] = feats
    return result


__all__ = [
    'extract_features', 'ensure_feature_vector', 'extract_and_predict'
]