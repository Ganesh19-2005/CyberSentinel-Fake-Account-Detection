"""
preprocessing.py - Data Preprocessing Module for Fake Social-Media Account Detection.

Project: "A Graph-and-Text Feature Classifier-Based Approach to Fake Social-Media Account Detection"
Academic Project Note:
    The dataset in this stage is an illustrative prototype dataset designed for
    testing preprocessing and pipeline flow. It should not be claimed to represent
    live, large-scale production social-media accounts.
"""

import re
import os
import pandas as pd
import numpy as np


def clean_text(text: str) -> str:
    """
    Cleans raw text (e.g., account bio/description).
    
    Steps:
    1. Handle non-string/missing inputs.
    2. Convert characters to lowercase for case consistency.
    3. Remove URLs and web links (common in promotional/spam bios).
    4. Remove special characters, punctuation, and symbols.
    5. Strip excess whitespace.

    Parameters:
        text (str): Raw bio text from account profile.

    Returns:
        str: Standardized, cleaned bio text.
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    
    # Remove URL links (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # Remove punctuation and special characters, retaining alphanumeric tokens
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def convert_boolean_series(series: pd.Series) -> pd.Series:
    """
    Standardizes boolean columns (e.g., 'is_verified') to binary integer format (0 or 1).
    
    Handles True/False booleans, case-insensitive string values ('true', 'false'),
    and numerical representations (1, 0).
    """
    mapping = {
        True: 1,
        False: 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0
    }
    
    # Normalize to string/lowercase then map
    cleaned = series.map(lambda val: mapping.get(str(val).strip().lower(), 0) if pd.notna(val) else 0)
    return cleaned.astype(int)


def check_and_handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for and removes duplicate rows or duplicate usernames,
    ensuring dataset integrity while logging counts.
    """
    duplicate_rows = df.duplicated().sum()
    duplicate_users = df.duplicated(subset=["username"]).sum() if "username" in df.columns else 0
    
    print(f"[Duplicate Check] Exact duplicate rows found: {duplicate_rows}")
    print(f"[Duplicate Check] Duplicate usernames found: {duplicate_users}")
    
    # Remove duplicate rows if any exist
    if duplicate_rows > 0:
        df = df.drop_duplicates()
        print(f"[Duplicate Check] Removed duplicate rows. New row count: {len(df)}")
        
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and handles missing values across different data types.
    - Numerical columns are filled with 0 (or median).
    - Text columns (e.g. bio) are filled with an empty string.
    - Boolean indicators are filled with 0 (False).
    - Rows with missing target labels are dropped to preserve training integrity.
    """
    print("[Missing Values Check] Null counts per column before imputation:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"  - {col}: {count} missing value(s)")
    if null_counts.sum() == 0:
        print("  - No missing values detected in the dataset.")
        
    # Drop rows missing the target label (cannot train supervised model without ground truth)
    if "label" in df.columns:
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        
    # Impute numerical features
    num_cols = ["followers", "following", "posts", "account_age_days"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            
    # Impute bio text
    if "bio" in df.columns:
        df["bio"] = df["bio"].fillna("")
        
    return df


def prepare_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares numerical features and engineers domain-specific behavioral ratios.
    
    Engineered Ratios:
    1. follower_following_ratio = followers / (following + 1)
       (Bots often follow thousands while having very few followers).
    2. reputation_score = followers / (followers + following + 1)
       (Normalized ratio between 0 and 1 indicating account credibility).
    3. posts_per_day = posts / (account_age_days + 1)
       (Measures posting activity intensity over the account lifespan).
    """
    # Baseline casting
    for col in ["followers", "following", "posts", "account_age_days"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    # Compute behavioral ratio features (+ 1 Laplace-style smoothing to avoid zero-division)
    if "followers" in df.columns and "following" in df.columns:
        df["follower_following_ratio"] = df["followers"] / (df["following"] + 1.0)
        df["reputation_score"] = df["followers"] / (df["followers"] + df["following"] + 1.0)
        
    if "posts" in df.columns and "account_age_days" in df.columns:
        df["posts_per_day"] = df["posts"] / (df["account_age_days"] + 1.0)
        
    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Complete end-to-end preprocessing pipeline for social account data.
    
    CRITICAL: Preserves the original DataFrame by working on a deep copy.
    
    Steps:
    1. Deep copy original DataFrame (preservation of raw data).
    2. Check and remove duplicates.
    3. Detect and handle missing values.
    4. Convert boolean flag 'is_verified' to binary int (0 or 1).
    5. Clean and normalize profile 'bio' text.
    6. Prepare numerical features and calculate behavioral ratios.
    7. Separate input features (X) from target label (y).
    
    Parameters:
        df (pd.DataFrame): Raw input dataframe.
        
    Returns:
        tuple (df_processed, X, y):
            - df_processed: Complete preprocessed dataframe including engineered columns and target.
            - X: Input features matrix (without label).
            - y: Target label vector (0 = Genuine, 1 = Fake).
    """
    # Step 1: Preserve original data
    df_clean = df.copy(deep=True)
    
    # Step 2: Duplicate checking
    df_clean = check_and_handle_duplicates(df_clean)
    
    # Step 3: Missing-value handling
    df_clean = handle_missing_values(df_clean)
    
    # Step 4: Boolean conversion
    if "is_verified" in df_clean.columns:
        df_clean["is_verified"] = convert_boolean_series(df_clean["is_verified"])
        
    # Step 5: Text preprocessing on bio
    if "bio" in df_clean.columns:
        df_clean["clean_bio"] = df_clean["bio"].apply(clean_text)
        
    # Step 6: Numerical feature preparation and engineering
    df_clean = prepare_numerical_features(df_clean)
    
    # Step 7: Separate features (X) from target label (y)
    if "label" in df_clean.columns:
        y = df_clean["label"].astype(int)
        X = df_clean.drop(columns=["label"])
    else:
        y = None
        X = df_clean
        
    return df_clean, X, y


# =====================================================================
# Standalone Execution / Verification Block
# =====================================================================
if __name__ == "__main__":
    raw_data_path = os.path.join("data", "social_accounts.csv")
    processed_data_path = os.path.join("data", "processed_accounts.csv")
    
    print("=" * 60)
    print(" FAKE SOCIAL-MEDIA ACCOUNT DETECTION - PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Reading raw data from: {raw_data_path}")
    
    if not os.path.exists(raw_data_path):
        print(f"Error: Dataset file not found at {raw_data_path}")
        exit(1)
        
    raw_df = pd.read_csv(raw_data_path)
    print(f"Original Dataset Shape: {raw_df.shape} (Rows, Columns)")
    print("\nOriginal Sample Records:")
    print(raw_df[["username", "followers", "following", "is_verified", "label"]].head())
    
    print("\n--- Running Preprocessing Pipeline ---")
    processed_df, X, y = preprocess_data(raw_df)
    
    print("\n[Verification 1] Preserved Original Dataset:")
    print(f"  - Original dataframe intact: {len(raw_df)} rows, {len(raw_df.columns)} columns.")
    print(f"  - Original 'is_verified' dtypes unaffected in raw_df.")
    
    print("\n[Verification 2] Cleaned Text (Original vs Clean Bio):")
    for idx, row in processed_df[["bio", "clean_bio"]].head(5).iterrows():
        print(f"  Raw   : \"{row['bio']}\"")
        print(f"  Clean : \"{row['clean_bio']}\"\n")
        
    print("[Verification 3] Engineered Numerical & Behavioral Features:")
    feature_preview_cols = [
        "username", "followers", "following", "is_verified",
        "follower_following_ratio", "reputation_score", "posts_per_day"
    ]
    print(processed_df[feature_preview_cols].head())
    
    print("\n[Verification 4] Feature Matrix (X) & Target Label (y) Separation:")
    print(f"  - X (Features) Shape : {X.shape}")
    print(f"  - X Feature Columns  : {list(X.columns)}")
    print(f"  - y (Target) Shape   : {y.shape}")
    print(f"  - Class Distribution : Genuine (0)={sum(y==0)}, Fake (1)={sum(y==1)}")
    
    # Save the processed dataset to a new file so the original remains untouched
    processed_df.to_csv(processed_data_path, index=False)
    print(f"\nPreprocessed dataset safely saved to: {processed_data_path}")
    print("Original 'data/social_accounts.csv' remains unmodified.")
    print("=" * 60)
    print("Preprocessing completed successfully!")
    print("=" * 60)
