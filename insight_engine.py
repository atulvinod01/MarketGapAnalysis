from __future__ import annotations

import re
import logging
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InsightEngineError(Exception):
    pass


class InsightEngine:
    
    SENTIMENT_LABELS: Dict[str, str] = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral", 
        "LABEL_2": "Positive"
    }
    
    def __init__(
        self,
        sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: int = -1
    ) -> None:
        self.sentiment_model: str = sentiment_model
        self._device: int = device
        self.sentiment_pipeline = None
        
        logger.info(f"Initializing InsightEngine with model: {sentiment_model}")
        
        try:
            import os
            import signal
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Model loading timed out")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            try:
                from transformers import pipeline
                
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=sentiment_model,
                    device=device,
                    truncation=True,
                    max_length=512
                )
                logger.info("Sentiment pipeline initialized successfully")
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        except TimeoutError:
            logger.warning("Model loading timed out after 30 seconds")
            logger.info("Using fast rule-based sentiment analysis instead")
            self.sentiment_pipeline = None
        except Exception as e:
            logger.warning(f"Failed to initialize HuggingFace model: {e}")
            logger.info("Using rule-based sentiment analysis")
            self.sentiment_pipeline = None
        
        self.tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer(
            ngram_range=(2, 2),
            stop_words="english",
            max_features=1000,
            min_df=2,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        logger.info("InsightEngine initialization complete")
    
    def _preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = text.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
        text = text.replace('&#x27;', "'").replace('&quot;', '"')
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _rule_based_sentiment(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        
        negative_words = [
            'hate', 'terrible', 'awful', 'worst', 'bad', 'horrible', 'sucks',
            'frustrating', 'annoying', 'broken', 'useless', 'disappointed',
            'waste', 'garbage', 'stupid', 'ridiculous', 'fail', 'never',
            'problem', 'issue', 'bug', 'crash', 'slow', 'expensive', 'overpriced'
        ]
        
        positive_words = [
            'love', 'great', 'amazing', 'awesome', 'excellent', 'fantastic',
            'wonderful', 'best', 'perfect', 'brilliant', 'good', 'nice',
            'helpful', 'useful', 'recommend', 'thanks', 'thank', 'appreciate'
        ]
        
        neg_count = sum(1 for w in negative_words if w in text_lower)
        pos_count = sum(1 for w in positive_words if w in text_lower)
        
        if neg_count > pos_count:
            confidence = min(0.5 + (neg_count * 0.1), 0.95)
            return "Negative", confidence
        elif pos_count > neg_count:
            confidence = min(0.5 + (pos_count * 0.1), 0.95)
            return "Positive", confidence
        else:
            return "Neutral", 0.6
    
    def get_negativity_index(self, df: pd.DataFrame) -> float:
        if df.empty or "sentiment_label" not in df.columns:
            return 0.0
        
        total = len(df)
        negative_count = len(df[df["sentiment_label"] == "Negative"])
        
        return round((negative_count / total) * 100, 1) if total > 0 else 0.0
    
    def _get_sentiment_score(self, label: str, score: float) -> float:
        if label == "Negative":
            return -score
        elif label == "Positive":
            return score
        else:
            return 0.0
    
    def analyze_sentiment(
        self,
        df: pd.DataFrame,
        text_column: str = "comment_text",
        batch_size: int = 32
    ) -> pd.DataFrame:
        if df.empty:
            logger.warning("Empty DataFrame provided for sentiment analysis")
            df = df.copy()
            df["sentiment_label"] = []
            df["sentiment_confidence"] = []
            df["sentiment_score"] = []
            df["preprocessed_text"] = []
            return df
        
        logger.info(f"Analyzing sentiment for {len(df)} comments...")
        
        preprocessed_texts: List[str] = [
            self._preprocess_text(str(text)) for text in df[text_column]
        ]
        
        valid_texts: List[str] = [
            text if text else "neutral" for text in preprocessed_texts
        ]
        
        sentiment_labels: List[str] = []
        sentiment_confidences: List[float] = []
        sentiment_scores: List[float] = []
        
        try:
            if self.sentiment_pipeline is not None:
                for i in range(0, len(valid_texts), batch_size):
                    batch = valid_texts[i:i + batch_size]
                    batch_results = self.sentiment_pipeline(batch)
                    
                    for result in batch_results:
                        raw_label = result["label"]
                        confidence = result["score"]
                        
                        if raw_label in ["negative", "NEGATIVE", "LABEL_0"]:
                            label = "Negative"
                        elif raw_label in ["positive", "POSITIVE", "LABEL_2"]:
                            label = "Positive"
                        else:
                            label = "Neutral"
                        
                        sentiment_labels.append(label)
                        sentiment_confidences.append(confidence)
                        sentiment_scores.append(self._get_sentiment_score(label, confidence))
                    
                    if (i + batch_size) % 100 == 0:
                        logger.info(f"Processed {min(i + batch_size, len(valid_texts))}/{len(valid_texts)} comments")
            else:
                logger.info("Using rule-based sentiment analysis")
                for text in valid_texts:
                    label, confidence = self._rule_based_sentiment(text)
                    sentiment_labels.append(label)
                    sentiment_confidences.append(confidence)
                    sentiment_scores.append(self._get_sentiment_score(label, confidence))
            
            df = df.copy()
            df["preprocessed_text"] = preprocessed_texts
            df["sentiment_label"] = sentiment_labels
            df["sentiment_confidence"] = sentiment_confidences
            df["sentiment_score"] = sentiment_scores
            
            label_counts = df["sentiment_label"].value_counts()
            logger.info(f"Sentiment distribution: {label_counts.to_dict()}")
            
            return df
            
        except Exception as e:
            raise InsightEngineError(f"Sentiment analysis failed: {e}")
    
    def extract_pain_points(
        self,
        df: pd.DataFrame,
        sentiment_column: str = "sentiment_label",
        text_column: str = "preprocessed_text",
        top_n: int = 20
    ) -> List[Tuple[str, float]]:
        negative_df = df[df[sentiment_column] == "Negative"].copy()
        
        if negative_df.empty:
            logger.warning("No negative comments found for pain point extraction")
            return []
        
        logger.info(f"Extracting pain points from {len(negative_df)} negative comments")
        
        texts: List[str] = negative_df[text_column].fillna("").tolist()
        texts = [t for t in texts if t.strip()]
        
        if len(texts) < 2:
            logger.warning("Not enough texts for TF-IDF analysis")
            return []
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            
            bigram_scores: List[Tuple[str, float]] = list(zip(feature_names, mean_scores))
            bigram_scores.sort(key=lambda x: x[1], reverse=True)
            
            top_bigrams = bigram_scores[:top_n]
            
            logger.info(f"Top pain points: {[b[0] for b in top_bigrams[:5]]}")
            
            return top_bigrams
            
        except Exception as e:
            logger.error(f"Pain point extraction failed: {e}")
            return []


if __name__ == "__main__":
    print("=" * 60)
    print("Insight Engine - Demo")
    print("=" * 60)
    
    sample_data = {
        "comment_text": [
            "I absolutely hate how complicated this software is!",
            "This product is amazing, solved all my problems!",
            "It's okay, nothing special really.",
            "The customer support is terrible, waited 3 hours!",
            "Great value for money, highly recommend!",
        ],
        "timestamp": ["2024-01-01"] * 5
    }
    
    df = pd.DataFrame(sample_data)
    
    try:
        engine = InsightEngine()
        df_analyzed = engine.analyze_sentiment(df)
        print("\nSentiment Analysis Results:")
        print(df_analyzed[["comment_text", "sentiment_label", "sentiment_score"]].to_string())
        
        print(f"\nNegativity Index: {engine.get_negativity_index(df_analyzed)}%")
        
    except InsightEngineError as e:
        print(f"Error: {e}")
