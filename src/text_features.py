"""
text_features.py - Text Feature Extraction Module for Fake Social-Media Account Detection.

Project: "A Graph-and-Text Feature Classifier-Based Approach to Fake Social-Media Account Detection"
Academic Note:
    Text features are extracted using Term Frequency-Inverse Document Frequency (TF-IDF).
    In this prototype phase, features are extracted from the illustrative dataset to validate
    pipeline components and serialization before moving to model training.
"""

import os
import sys
import re
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import text cleaning logic from preprocessing module
try:
    from src.preprocessing import clean_text
except ImportError:
    # Standalone fallback if invoked outside package scope
    def clean_text(text: str) -> str:
        if pd.isna(text) or text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()


def clean_username_for_nlp(username: str) -> str:
    """
    Normalizes a social media username into tokenizable text.
    For example:
        'tech_raj' -> 'tech raj'
        'promo_bot_45' -> 'promo bot 45'
        'free_gift_777' -> 'free gift 777'
    This allows NLP vectorizers to capture spam/bot cues embedded inside account handles.
    """
    if pd.isna(username) or username is None:
        return ""
    # Replace underscores, hyphens, and dots with spaces
    cleaned = re.sub(r"[_\-\.]", " ", str(username).lower())
    # Separate alphabetic and numeric blocks gently (e.g. user928 -> user 928)
    cleaned = re.sub(r"([a-zA-Z]+)(\d+)", r"\1 \2", cleaned)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def prepare_combined_text_corpus(df: pd.DataFrame) -> pd.Series:
    """
    Combines cleaned username and bio text for each account into a unified text document.
    
    Why combine username and bio?
    Fake accounts often place spam keywords ('crypto', 'promo', 'bot', 'free') directly in their handles
    or split promotional pitches across their handle and bio. Combining them provides a single, rich
    text profile for each user.
    
    Parameters:
        df (pd.DataFrame): Accounts dataframe containing 'username' and 'bio' (or 'clean_bio').
        
    Returns:
        pd.Series: Series of combined text strings, aligned with df's index.
    """
    # 1. Clean username
    if "username" in df.columns:
        clean_usernames = df["username"].apply(clean_username_for_nlp)
    else:
        clean_usernames = pd.Series([""] * len(df), index=df.index)
        
    # 2. Extract or clean bio
    if "clean_bio" in df.columns:
        clean_bios = df["clean_bio"].fillna("").astype(str)
    elif "bio" in df.columns:
        clean_bios = df["bio"].apply(clean_text)
    else:
        clean_bios = pd.Series([""] * len(df), index=df.index)
        
    # 3. Concatenate username and bio
    combined = (clean_usernames + " " + clean_bios).str.strip()
    # Replace any empty strings with placeholder token to avoid empty document errors in vectorizers
    combined = combined.replace("", "empty")
    
    return combined


def fit_and_save_tfidf(
    corpus: pd.Series,
    max_features: int = 25,
    model_save_path: str = os.path.join("models", "tfidf_vectorizer.pkl")
) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Fits a Scikit-Learn TfidfVectorizer on the training text corpus, saves the fitted
    vectorizer to disk, and returns the TF-IDF feature DataFrame.
    
    Parameters:
        corpus (pd.Series): Text documents (combined username + bio).
        max_features (int): Top n-gram features to retain (prevents high-dimensional curse).
        model_save_path (str): Destination path to serialize the fitted vectorizer.
        
    Returns:
        tuple (tfidf_df, vectorizer):
            - tfidf_df (pd.DataFrame): DataFrame containing TF-IDF features with descriptive columns.
            - vectorizer (TfidfVectorizer): The fitted Scikit-learn vectorizer object.
    """
    # Initialize TF-IDF Vectorizer
    # min_df=1 ensures terms in small prototype datasets are captured
    # stop_words='english' removes uninformative filler words
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=1,
        ngram_range=(1, 2),
        lowercase=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = [f"tfidf_{name.replace(' ', '_')}" for name in vectorizer.get_feature_names_out()]
    
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=corpus.index
    )
    
    # Save the fitted vectorizer so it can be re-used during live inference/prediction
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(vectorizer, model_save_path)
        print(f"[Text Features] Fitted TF-IDF Vectorizer saved to: {model_save_path}")
        
    return tfidf_df, vectorizer


def transform_with_saved_tfidf(
    corpus: pd.Series,
    vectorizer_or_path
) -> pd.DataFrame:
    """
    Transforms new account text using a previously fitted TF-IDF Vectorizer.
    This guarantees feature alignment without data leakage.
    
    Parameters:
        corpus (pd.Series): Series of text strings for new/unseen accounts.
        vectorizer_or_path: An in-memory fitted TfidfVectorizer or a string filepath to a .pkl.
        
    Returns:
        pd.DataFrame: TF-IDF feature matrix with matching column names.
    """
    if isinstance(vectorizer_or_path, str):
        vectorizer = joblib.load(vectorizer_or_path)
    else:
        vectorizer = vectorizer_or_path
        
    tfidf_matrix = vectorizer.transform(corpus)
    feature_names = [f"tfidf_{name.replace(' ', '_')}" for name in vectorizer.get_feature_names_out()]
    
    return pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=corpus.index
    )


def extract_text_features(
    df: pd.DataFrame,
    max_features: int = 25,
    save_vectorizer: bool = True,
    model_save_path: str = os.path.join("models", "tfidf_vectorizer.pkl")
) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    High-level entry point to extract text features from an account DataFrame.
    
    Parameters:
        df (pd.DataFrame): Account dataframe.
        max_features (int): Maximum vocabulary size for TF-IDF.
        save_vectorizer (bool): Whether to persist the fitted vectorizer to disk.
        model_save_path (str): Filepath to save the vectorizer.
        
    Returns:
        tuple (tfidf_features_df, fitted_vectorizer)
    """
    corpus = prepare_combined_text_corpus(df)
    save_path = model_save_path if save_vectorizer else None
    tfidf_df, vectorizer = fit_and_save_tfidf(corpus, max_features=max_features, model_save_path=save_path)
    return tfidf_df, vectorizer


# =====================================================================
# Standalone Execution / Verification Block
# =====================================================================
if __name__ == "__main__":
    data_path = os.path.join("data", "processed_accounts.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "social_accounts.csv")
        
    print("=" * 65)
    print(" TASK 1: TEXT FEATURE EXTRACTION (TF-IDF & USERNAME-BIO FUSION)")
    print("=" * 65)
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # 1. Prepare combined corpus
    corpus = prepare_combined_text_corpus(df)
    print("\n[Sample Combined Text Representations]:")
    for username, doc in zip(df["username"].head(5), corpus.head(5)):
        print(f"  Account '@{username}': \"{doc}\"")
        
    # 2. Fit TF-IDF and serialize
    tfidf_features, vectorizer = extract_text_features(df, max_features=20, save_vectorizer=True)
    
    print("\n[TF-IDF Vocabulary Extracted]:")
    vocab = vectorizer.get_feature_names_out()
    print(f"  Total Vocabulary Size: {len(vocab)}")
    print(f"  Features: {list(vocab)}")
    
    print("\n[Sample TF-IDF Feature Vectors (First 3 Accounts)]:")
    print(tfidf_features.head(3))
    
    # 3. Test serialization & inference reload
    model_path = os.path.join("models", "tfidf_vectorizer.pkl")
    print(f"\n[Verification] Testing reload from saved model file ({model_path}):")
    loaded_vectorizer = joblib.load(model_path)
    test_sample = pd.Series(["crypto bot double money win prizes", "tech raj coding in python"])
    transformed_test = transform_with_saved_tfidf(test_sample, loaded_vectorizer)
    print("  Successfully transformed new sample text without retraining!")
    print(f"  Transformed Shape: {transformed_test.shape}")
    print("=" * 65)
    print("Text feature extraction completed successfully!")
    print("=" * 65)
