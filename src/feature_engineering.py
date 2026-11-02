"""
feature_engineering.py - Feature Integration and Leakage Prevention Module.

Project: "A Graph-and-Text Feature Classifier-Based Approach to Fake Social-Media Account Detection"
Academic Project Note:
    Combines profile numerical features, NLP text TF-IDF features, and NetworkX graph structural
    features into an integrated feature representation.
    Ensures strict separation between input features (X) and ground-truth target (y) to prevent
    data leakage. The data used here is an illustrative prototype for engineering validation.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on sys.path for both direct script execution and module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_data
from src.text_features import extract_text_features
from src.graph_features import build_account_graph, extract_graph_features


def integrate_all_features(
    accounts_csv: str = os.path.join("data", "social_accounts.csv"),
    edges_csv: str = os.path.join("data", "illustrative_account_edges.csv"),
    tfidf_max_features: int = 20,
    save_vectorizer: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Orchestrates end-to-end feature extraction and integrates all modalities:
    1. Preprocessed Numerical & Behavioral Features (Profile-level)
    2. TF-IDF Text Features (Handle & Bio NLP-level)
    3. Directed NetworkX Graph Features (Topology-level)
    
    Prevents Target Leakage:
    - Extracts target 'y' immediately and drops it from the feature pool.
    - None of the text, numerical, or graph transformations consume or rely on 'label'.
    
    Parameters:
        accounts_csv (str): Path to raw accounts CSV.
        edges_csv (str): Path to relationship edge-list CSV.
        tfidf_max_features (int): Top TF-IDF vocabulary size.
        save_vectorizer (bool): Whether to persist fitted TF-IDF vectorizer to models/.
        
    Returns:
        tuple (integrated_df, X, y):
            - integrated_df (pd.DataFrame): Complete merged dataset including identifier and target.
            - X (pd.DataFrame): Pure numeric/engineered feature matrix ready for ML estimators.
            - y (pd.Series): Binary target labels (0 = Genuine, 1 = Fake).
    """
    print("[Integration Step 1] Loading and preprocessing account dataset...")
    raw_df = pd.read_csv(accounts_csv)
    preprocessed_df, _, raw_y = preprocess_data(raw_df)
    
    # Isolate ground-truth target label to prevent any data leakage
    y = preprocessed_df["label"].copy().astype(int)
    
    print("[Integration Step 2] Extracting TF-IDF text features from usernames and bios...")
    # Extract TF-IDF text features
    tfidf_df, _ = extract_text_features(
        preprocessed_df,
        max_features=tfidf_max_features,
        save_vectorizer=save_vectorizer,
        model_save_path=os.path.join("models", "tfidf_vectorizer.pkl")
    )
    
    print("[Integration Step 3] Extracting NetworkX topological graph features...")
    # Build NetworkX graph and compute structural metrics
    usernames = list(preprocessed_df["username"])
    G = build_account_graph(usernames, edge_list_path=edges_csv)
    graph_df = extract_graph_features(G, usernames)
    
    print("[Integration Step 4] Merging all feature modalities by account identifier...")
    # Baseline numerical & behavioral columns
    numerical_cols = [
        "followers", "following", "posts", "account_age_days",
        "is_verified", "follower_following_ratio", "reputation_score", "posts_per_day"
    ]
    
    # 1. Start with username, label, and numerical features
    base_df = preprocessed_df[["username"] + numerical_cols].copy()
    
    # 2. Merge TF-IDF text features (aligned on index)
    combined_df = pd.concat([base_df, tfidf_df], axis=1)
    
    # 3. Merge Graph features on 'username'
    integrated_df = pd.merge(combined_df, graph_df, on="username", how="left")
    
    # 4. Attach ground truth label at the end for inspection/storage
    integrated_df["label"] = y.values
    
    # Ensure no missing values were introduced during merges
    assert integrated_df.isnull().sum().sum() == 0, "Missing values detected in integrated feature table!"
    
    # Define pure feature matrix X (drop identifiers, raw text, and target labels)
    non_feature_cols = ["username", "label"]
    feature_cols = [col for col in integrated_df.columns if col not in non_feature_cols]
    X = integrated_df[feature_cols].copy()
    
    return integrated_df, X, y


def save_feature_artifacts(
    integrated_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str = "outputs"
) -> None:
    """
    Saves the extracted feature matrices and structural metadata to the outputs/ directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    integrated_path = os.path.join(output_dir, "integrated_dataset.csv")
    x_path = os.path.join(output_dir, "feature_matrix_X.csv")
    y_path = os.path.join(output_dir, "target_labels_y.csv")
    summary_path = os.path.join(output_dir, "features_summary.json")
    
    integrated_df.to_csv(integrated_path, index=False)
    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False, header=["label"])
    
    # Categorize features for transparent reporting
    numerical_features = [col for col in X.columns if not col.startswith("tfidf_") and not col.startswith("graph_")]
    tfidf_features = [col for col in X.columns if col.startswith("tfidf_")]
    graph_features = [col for col in X.columns if col.startswith("graph_")]
    
    summary = {
        "total_samples": int(len(X)),
        "total_features": int(X.shape[1]),
        "feature_breakdown": {
            "numerical_and_behavioral_count": len(numerical_features),
            "numerical_and_behavioral_features": numerical_features,
            "tfidf_text_count": len(tfidf_features),
            "tfidf_text_features": tfidf_features,
            "graph_structural_count": len(graph_features),
            "graph_structural_features": graph_features
        },
        "target_distribution": {
            "genuine_count_0": int(sum(y == 0)),
            "fake_count_1": int(sum(y == 1))
        }
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n[Artifacts Saved in '{output_dir}/']:")
    print(f"  - Full Aligned Table : {integrated_path}")
    print(f"  - Feature Matrix (X) : {x_path}")
    print(f"  - Target Vector (y)  : {y_path}")
    print(f"  - Feature Summary    : {summary_path}")


# =====================================================================
# Standalone Execution / Verification Block
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" TASK 3 & 4: FEATURE INTEGRATION, LEAKAGE CHECK & TESTING")
    print("=" * 65)
    
    integrated_df, X, y = integrate_all_features(
        accounts_csv=os.path.join("data", "social_accounts.csv"),
        edges_csv=os.path.join("data", "illustrative_account_edges.csv"),
        tfidf_max_features=20,
        save_vectorizer=True
    )
    
    print("\n--- SANITY & DIMENSIONALITY CHECKS ---")
    print(f"Total Accounts Processed : {len(integrated_df)}")
    print(f"Feature Matrix X Shape   : {X.shape} (Rows, Features)")
    print(f"Target Vector y Shape    : {y.shape}")
    print(f"Null Values in X         : {X.isnull().sum().sum()}")
    print(f"Null Values in y         : {y.isnull().sum()}")
    print(f"Target Label Leakage Check: {'Passed (label absent from X)' if 'label' not in X.columns else 'FAILED'}")
    
    save_feature_artifacts(integrated_df, X, y, output_dir="outputs")
    
    print("\n[Preview of Extracted Feature Columns]:")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i:2d}. {col}")
        
    print("=" * 65)
    print("Feature integration phase completed successfully!")
    print("=" * 65)
