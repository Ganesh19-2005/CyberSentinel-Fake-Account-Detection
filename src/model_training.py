"""
model_training.py - Model Training, Evaluation, and Leakage-Free Pipeline.

Project: "A Graph-and-Text Feature Classifier-Based Approach to Fake Social-Media Account Detection"
Academic Note:
    This module trains, evaluates, and compares three supervised classifiers:
    1. Logistic Regression
    2. Random Forest Classifier
    3. Support Vector Machine (SVM with RBF kernel)
    
    Data Leakage Prevention:
    - Train/Test splitting is performed BEFORE feature scaling and TF-IDF fitting.
    - TF-IDF vocabulary and StandardScaler parameters are learned strictly from X_train.
    - Graph structural features are computed from relationship topology with zero label awareness.
    
    Important Research Disclaimer:
    - The current 10-record dataset is an illustrative prototype. High metrics on this sample
      reflect the simplicity and separability of the prototype, NOT real-world generalized accuracy.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure non-interactive Matplotlib backend and temporary config directory
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_fake_acc")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.preprocessing import preprocess_data
from src.text_features import prepare_combined_text_corpus
from src.graph_features import build_account_graph, extract_graph_features


def prepare_leakage_free_train_test_splits(
    accounts_csv: str = os.path.join("data", "social_accounts.csv"),
    edges_csv: str = os.path.join("data", "illustrative_account_edges.csv"),
    test_size: float = 0.3,
    random_state: int = 42,
    tfidf_max_features: int = 15
) -> dict:
    """
    Performs data splitting and transforms features with zero data leakage.
    
    Leakage Prevention Protocol:
    1. Preprocesses raw records without using target labels.
    2. Extracts graph structural features from edge topology (no target awareness).
    3. Combines features and splits into Train and Test subsets using stratified sampling.
    4. Fits TfidfVectorizer strictly on train_df['combined_text'], then transforms test_df.
    5. Fits StandardScaler strictly on train_df[numerical_cols], then transforms test_df.
    6. Returns aligned X_train, X_test, y_train, y_test, scaler, vectorizer, and metadata.
    """
    print("[Pipeline Step 1] Loading and preprocessing accounts data...")
    raw_df = pd.read_csv(accounts_csv)
    df_clean, _, _ = preprocess_data(raw_df)
    
    print("[Pipeline Step 2] Extracting graph features from network edge list...")
    usernames = list(df_clean["username"])
    G = build_account_graph(usernames, edge_list_path=edges_csv)
    graph_df = extract_graph_features(G, usernames)
    
    # Merge preprocessed tabular data with graph features
    merged_df = pd.merge(df_clean, graph_df, on="username", how="left")
    
    # Generate combined username + bio text representations
    merged_df["combined_text"] = prepare_combined_text_corpus(merged_df)
    
    # Identify numerical feature columns (Profile + Graph features)
    numerical_cols = [
        "followers", "following", "posts", "account_age_days",
        "is_verified", "follower_following_ratio", "reputation_score", "posts_per_day",
        "graph_in_degree", "graph_out_degree", "graph_total_degree", "graph_in_out_ratio",
        "graph_pagerank", "graph_betweenness", "graph_clustering_coefficient",
        "graph_weak_component_size", "graph_strong_component_size"
    ]
    
    print("[Pipeline Step 3] Performing Stratified Train-Test Split (test_size={})...".format(test_size))
    # Stratified split to ensure both classes (genuine=0, fake=1) are represented in train and test sets
    train_df, test_df = train_test_split(
        merged_df,
        test_size=test_size,
        random_state=random_state,
        stratify=merged_df["label"]
    )
    
    y_train = train_df["label"].astype(int).reset_index(drop=True)
    y_test = test_df["label"].astype(int).reset_index(drop=True)
    
    print(f"  - Training samples : {len(train_df)} (Genuine: {sum(y_train==0)}, Fake: {sum(y_train==1)})")
    print(f"  - Testing samples  : {len(test_df)} (Genuine: {sum(y_test==0)}, Fake: {sum(y_test==1)})")
    
    print("[Pipeline Step 4] Fitting TF-IDF Vectorizer on Train Split ONLY...")
    # Strict Leakage Prevention: fit vocabulary ONLY on train split
    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        stop_words="english",
        min_df=1,
        ngram_range=(1, 2)
    )
    train_tfidf_matrix = vectorizer.fit_transform(train_df["combined_text"]).toarray()
    test_tfidf_matrix = vectorizer.transform(test_df["combined_text"]).toarray()
    
    tfidf_cols = [f"tfidf_{name.replace(' ', '_')}" for name in vectorizer.get_feature_names_out()]
    train_tfidf_df = pd.DataFrame(train_tfidf_matrix, columns=tfidf_cols)
    test_tfidf_df = pd.DataFrame(test_tfidf_matrix, columns=tfidf_cols)
    
    print("[Pipeline Step 5] Fitting StandardScaler on Train Split ONLY...")
    # Strict Leakage Prevention: learn mean and variance ONLY from train split
    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(train_df[numerical_cols])
    test_num_scaled = scaler.transform(test_df[numerical_cols])
    
    train_num_df = pd.DataFrame(train_num_scaled, columns=[f"scaled_{c}" for c in numerical_cols])
    test_num_df = pd.DataFrame(test_num_scaled, columns=[f"scaled_{c}" for c in numerical_cols])
    
    # Concatenate numerical and text features
    X_train = pd.concat([train_num_df, train_tfidf_df], axis=1)
    X_test = pd.concat([test_num_df, test_tfidf_df], axis=1)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_usernames": list(train_df["username"]),
        "test_usernames": list(test_df["username"]),
        "scaler": scaler,
        "vectorizer": vectorizer,
        "feature_names": list(X_train.columns)
    }


def train_and_evaluate_models(data_splits: dict) -> dict:
    """
    Trains Logistic Regression, Random Forest, and SVM classifiers.
    Calculates Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    X_train = data_splits["X_train"]
    X_test = data_splits["X_test"]
    y_train = data_splits["y_train"]
    y_test = data_splits["y_test"]
    
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, solver="lbfgs", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
        "Support Vector Machine (SVM)": SVC(kernel="rbf", C=1.0, random_state=42)
    }
    
    results = {}
    
    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train, y_train)
        
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        acc = float(accuracy_score(y_test, y_test_pred))
        prec = float(precision_score(y_test, y_test_pred, zero_division=0))
        rec = float(recall_score(y_test, y_test_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).tolist()
        
        report_str = classification_report(
            y_test,
            y_test_pred,
            labels=[0, 1],
            target_names=["Genuine (0)", "Fake (1)"],
            zero_division=0
        )
        
        results[name] = {
            "model_obj": clf,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report_str,
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "y_pred": y_test_pred.tolist()
        }
        
        print(f"  Test Accuracy  : {acc:.4f}")
        print(f"  Precision      : {prec:.4f}")
        print(f"  Recall         : {rec:.4f}")
        print(f"  F1-Score       : {f1:.4f}")
        print(f"  Confusion Matrix [TN, FP / FN, TP]: {cm}")
        print("\nClassification Report:\n" + report_str)
        
    return results


def save_models_and_transformers(
    models_dict: dict,
    scaler: StandardScaler,
    vectorizer: TfidfVectorizer,
    models_dir: str = "models"
) -> str:
    """
    Saves trained models, preprocessing objects, and the best-performing model to disk.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Save preprocessing transformers
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    print(f"\n[Models Saved] Preprocessing transformers saved to '{models_dir}/'")
    
    filename_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "Support Vector Machine (SVM)": "svm_classifier.pkl"
    }
    
    best_name = None
    best_score = -1.0
    
    for name, data in models_dict.items():
        fname = filename_map[name]
        path = os.path.join(models_dir, fname)
        joblib.dump(data["model_obj"], path)
        print(f"  - {name} saved to: {path}")
        
        # Select best model based on F1-score (with accuracy as secondary tiebreaker)
        score = data["f1_score"] * 10.0 + data["accuracy"]
        if score > best_score:
            best_score = score
            best_name = name
            
    # Save best model copy
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(models_dict[best_name]["model_obj"], best_model_path)
    print(f"  - [Best Model] '{best_name}' serialized as default at: {best_model_path}")
    
    return best_name


def generate_visualizations(
    results: dict,
    output_dir: str = "outputs"
) -> list[str]:
    """
    Generates academic publication-style plots:
    1. Model Performance Comparison (Bar Chart)
    2. Confusion Matrices (Heatmaps)
    3. Precision, Recall & F1 Trade-off Comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    
    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in model_names]
    precisions = [results[m]["precision"] for m in model_names]
    recalls = [results[m]["recall"] for m in model_names]
    f1_scores = [results[m]["f1_score"] for m in model_names]
    
    # -------------------------------------------------------------
    # Plot 1: Overall Performance Comparison (Grouped Bar Chart)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(model_names))
    width = 0.2
    
    ax.bar(x - 1.5 * width, accuracies, width, label="Accuracy", color="#2b5c8f")
    ax.bar(x - 0.5 * width, precisions, width, label="Precision", color="#41924B")
    ax.bar(x + 0.5 * width, recalls, width, label="Recall", color="#e67e22")
    ax.bar(x + 1.5 * width, f1_scores, width, label="F1-Score", color="#8e44ad")
    
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Classifier Performance Comparison (Illustrative Benchmark)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (SVM)", "") for m in model_names], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Add values on top of bars
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "model_performance_comparison.png")
    fig.savefig(chart1_path, dpi=300)
    plt.close(fig)
    saved_plots.append(chart1_path)
    print(f"[Visualizations] Saved Performance Comparison to: {chart1_path}")
    
    # -------------------------------------------------------------
    # Plot 2: Confusion Matrix Heatmaps (1 x 3 Subplots)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    cm_labels = ["Genuine (0)", "Fake (1)"]
    
    for idx, (name, data) in enumerate(results.items()):
        cm = np.array(data["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=cm_labels,
            yticklabels=cm_labels,
            annot_kws={"size": 13, "weight": "bold"}
        )
        axes[idx].set_title(f"{name}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("True Label" if idx == 0 else "", fontsize=10)
        
    plt.suptitle("Confusion Matrix Heatmaps on Test Set", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "confusion_matrices.png")
    fig.savefig(chart2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(chart2_path)
    print(f"[Visualizations] Saved Confusion Matrix Heatmaps to: {chart2_path}")
    
    # -------------------------------------------------------------
    # Plot 3: Precision-Recall-F1 Detailed Breakdown
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics_df = pd.DataFrame({
        "Model": [m.replace(" (SVM)", "") for m in model_names],
        "Precision": precisions,
        "Recall": recalls,
        "F1-Score": f1_scores
    }).melt(id_vars="Model", var_name="Metric", value_name="Value")
    
    sns.barplot(data=metrics_df, x="Metric", y="Value", hue="Model", palette="Set2", ax=ax)
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision, Recall, and F1-Score Trade-off", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
        
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "precision_recall_f1_comparison.png")
    fig.savefig(chart3_path, dpi=300)
    plt.close(fig)
    saved_plots.append(chart3_path)
    print(f"[Visualizations] Saved Detailed Metrics Comparison to: {chart3_path}")
    
    return saved_plots


def export_evaluation_report(results: dict, best_model_name: str, output_dir: str = "outputs"):
    """
    Exports a structured JSON summary of evaluation metrics and experimental notes.
    """
    clean_results = {}
    for name, data in results.items():
        clean_results[name] = {
            "accuracy": data["accuracy"],
            "precision": data["precision"],
            "recall": data["recall"],
            "f1_score": data["f1_score"],
            "confusion_matrix": data["confusion_matrix"],
            "train_accuracy": data["train_accuracy"]
        }
        
    report = {
        "best_model": best_model_name,
        "results_summary": clean_results,
        "academic_limitations_and_notes": {
            "dataset_nature": "Illustrative prototype benchmark with 10 records (7 train, 3 test).",
            "real_world_applicability": (
                "The observed high scores are a mathematical artifact of the small, cleanly separable "
                "synthetic dataset. These numbers must NOT be cited as real-world predictive performance."
            ),
            "recommendation": (
                "For B.Tech project thesis defense, validate the architecture on public research benchmarks "
                "such as Cresci-2015/2017 or TwiBot-20/22."
            )
        }
    }
    
    out_path = os.path.join(output_dir, "model_evaluation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\n[Report Exported] Evaluation metrics written to: {out_path}")


# =====================================================================
# Main Execution Block
# =====================================================================
if __name__ == "__main__":
    print("=" * 68)
    print(" PHASE 3: MODEL TRAINING, EVALUATION & LEAKAGE-FREE PIPELINE")
    print("=" * 68)
    
    # 1. Prepare Leakage-Free Train and Test Splits
    splits = prepare_leakage_free_train_test_splits(
        accounts_csv=os.path.join("data", "social_accounts.csv"),
        edges_csv=os.path.join("data", "illustrative_account_edges.csv"),
        test_size=0.3,
        random_state=42,
        tfidf_max_features=15
    )
    
    # 2. Train and Evaluate Classifiers
    eval_results = train_and_evaluate_models(splits)
    
    # 3. Save Models and Preprocessing Objects
    best_clf = save_models_and_transformers(
        eval_results,
        scaler=splits["scaler"],
        vectorizer=splits["vectorizer"],
        models_dir="models"
    )
    
    # 4. Generate Visualizations
    plots = generate_visualizations(eval_results, output_dir="outputs")
    
    # 5. Export JSON Report
    export_evaluation_report(eval_results, best_model_name=best_clf, output_dir="outputs")
    
    print("\n" + "=" * 68)
    print(f" PIPELINE FINISHED! Best Performing Model: {best_clf}")
    print("=" * 68)
