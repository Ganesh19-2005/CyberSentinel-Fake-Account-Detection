"""
app.py - CyberSentinel: Fake Social-Media Account Detection
Streamlit Cybersecurity Command-Center Web Application.

Project: "A Graph-and-Text Feature Classifier-Based Approach to Fake Social-Media Account Detection"
Academic Note:
    Multimodal classification pipeline integrating Profile Analytics,
    NLP TF-IDF Linguistic Features, and NetworkX Graph Topology.
"""

import os
import sys
import json
import textwrap
import joblib
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import reusable preprocessing and feature extraction logic
from src.preprocessing import clean_text
from src.text_features import clean_username_for_nlp
from src.graph_features import build_account_graph, extract_graph_features


# =====================================================================
# HTML / CSS Rendering Helper (Guarantees Zero Raw Code Display)
# =====================================================================
def render_html(html_code: str, in_sidebar: bool = False):
    """
    Renders HTML cleanly into Streamlit DOM without Markdown indentation code-block traps.
    Uses st.html / st.sidebar.html if available, or dedented st.markdown.
    """
    clean_html = textwrap.dedent(html_code).strip()
    target = st.sidebar if in_sidebar else st
    if hasattr(target, "html"):
        target.html(clean_html)
    else:
        target.markdown(clean_html, unsafe_allow_html=True)


# =====================================================================
# Page Configuration & Cybersecurity Theming (Red & Black Palette)
# =====================================================================
st.set_page_config(
    page_title="CYBERSENTINEL | Threat Detection System",
    page_icon="⛨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust, High-Contrast SVG Cybersecurity Shield Logo (No external dependencies)
CYBERSENTINEL_SHIELD_SVG = (
    '<svg width="56" height="56" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0 auto;">'
    '<path d="M50 6 L88 22 V50 C88 74 71.5 91.5 50 96 C28.5 91.5 12 74 12 50 V22 L50 6 Z" fill="#121212" stroke="#E50914" stroke-width="4"/>'
    '<path d="M50 15 L80 27 V50 C80 69 66.5 83.5 50 88 C33.5 83.5 20 69 20 50 V27 L50 15 Z" fill="#191919" stroke="#8B0000" stroke-width="2"/>'
    '<path d="M50 28 L69 37 V50 C69 62 60 71 50 74 C40 71 31 62 31 50 V37 L50 28 Z" fill="#8B0000" stroke="#E50914" stroke-width="2.5"/>'
    '<line x1="50" y1="36" x2="50" y2="64" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>'
    '<line x1="36" y1="50" x2="64" y2="50" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>'
    '<circle cx="50" cy="50" r="5" fill="#E50914" stroke="#FFFFFF" stroke-width="1.8"/>'
    '</svg>'
)


def apply_cybersecurity_theme():
    """Injects high-end matte black and crimson red cybersecurity CSS."""
    theme_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-main: #080808;
        --bg-secondary: #121212;
        --bg-card: #191919;
        --accent-red: #E50914;
        --accent-dark-red: #8B0000;
        --text-main: #FFFFFF;
        --text-muted: #B3B3B3;
        --border-red: #441414;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(139, 0, 0, 0.12) 0%, transparent 45%),
            radial-gradient(circle at 85% 85%, rgba(229, 9, 20, 0.08) 0%, transparent 45%) !important;
        color: var(--text-main) !important;
        font-family: 'Inter', sans-serif !important;
    }

    h1, h2, h3 {
        font-family: 'Orbitron', 'Rajdhani', sans-serif !important;
        letter-spacing: 0.05em !important;
        color: var(--text-main) !important;
    }
    h4, h5, h6 {
        font-family: 'Rajdhani', 'Inter', sans-serif !important;
        letter-spacing: 0.04em !important;
        color: #E2E8F0 !important;
    }

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #080808;
    }
    ::-webkit-scrollbar-thumb {
        background: #441414;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #E50914;
    }

    section[data-testid="stSidebar"] {
        background-color: #0c0c0c !important;
        border-right: 1px solid var(--border-red) !important;
        box-shadow: 4px 0 25px rgba(0, 0, 0, 0.8) !important;
    }

    div[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        background: #141414 !important;
        border: 1px solid var(--border-red) !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        margin-bottom: 8px !important;
        color: var(--text-muted) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
    }

    div[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: var(--accent-red) !important;
        background: rgba(229, 9, 20, 0.12) !important;
        color: #FFFFFF !important;
        transform: translateX(3px) !important;
    }

    div[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(229, 9, 20, 0.3) 0%, rgba(139, 0, 0, 0.15) 100%) !important;
        border-left: 4px solid var(--accent-red) !important;
        border-color: var(--accent-red) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.25) !important;
    }

    .cyber-card {
        background: linear-gradient(145deg, #191919 0%, #121212 100%);
        border: 1px solid var(--border-red);
        border-radius: 10px;
        padding: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        margin-bottom: 16px;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .cyber-card:hover {
        border-color: var(--accent-red);
        box-shadow: 0 6px 24px rgba(229, 9, 20, 0.25);
        transform: translateY(-2px);
    }
    .cyber-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, var(--accent-red) 0%, var(--accent-dark-red) 100%);
    }

    .metric-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 6px;
        font-weight: 600;
    }

    .metric-val {
        font-family: 'Orbitron', monospace;
        font-size: 2.1rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.15);
    }

    .metric-badge {
        display: inline-block;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 700;
        margin-top: 8px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-genuine {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .badge-fake {
        background-color: rgba(229, 9, 20, 0.18);
        color: #E50914;
        border: 1px solid rgba(229, 9, 20, 0.45);
    }

    .badge-cyber {
        background-color: rgba(139, 0, 0, 0.25);
        color: #FFA3A3;
        border: 1px solid #8B0000;
    }

    .alert-card-fake {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.2) 0%, rgba(139, 0, 0, 0.1) 100%);
        border: 1px solid #E50914;
        border-left: 6px solid #E50914;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 0 25px rgba(229, 9, 20, 0.25);
    }

    .alert-card-genuine {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 78, 54, 0.1) 100%);
        border: 1px solid #10B981;
        border-left: 6px solid #10B981;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
    }

    .info-pill {
        background: #141414;
        border: 1px solid var(--border-red);
        border-left: 4px solid var(--accent-red);
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.92rem;
        margin: 14px 0;
        color: #D1D5DB;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextArea"] textarea {
        background-color: #121212 !important;
        border: 1px solid var(--border-red) !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent-red) !important;
        box-shadow: 0 0 10px rgba(229, 9, 20, 0.3) !important;
    }

    div[data-testid="stFormSubmitButton"] > button,
    div.stButton > button {
        background: linear-gradient(135deg, #E50914 0%, #8B0000 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #E50914 !important;
        font-family: 'Rajdhani', 'Orbitron', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        border-radius: 6px !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.35) !important;
        transition: all 0.25s ease !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover,
    div.stButton > button:hover {
        background: linear-gradient(135deg, #FF1F2D 0%, #B30000 100%) !important;
        box-shadow: 0 6px 25px rgba(229, 9, 20, 0.6) !important;
        border-color: #FFFFFF !important;
        transform: translateY(-2px) !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #121212 !important;
        border-color: var(--border-red) !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-red) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    </style>
    """
    render_html(theme_css)


# =====================================================================
# Data & Model Loader
# =====================================================================
@st.cache_resource
def load_models_and_artifacts():
    """Loads serialized models, vectorizers, and evaluation metrics with graceful fallback."""
    models_dir = Path("models")
    outputs_dir = Path("outputs")
    
    artifacts = {
        "scaler": None,
        "vectorizer": None,
        "best_model": None,
        "models": {},
        "metrics": None,
        "is_ready": False,
        "missing_files": []
    }
    
    required_files = {
        "scaler": models_dir / "scaler.pkl",
        "vectorizer": models_dir / "tfidf_vectorizer.pkl",
        "best_model": models_dir / "best_model.pkl"
    }
    
    for key, path in required_files.items():
        if path.exists():
            artifacts[key] = joblib.load(path)
        else:
            artifacts["missing_files"].append(str(path))
            
    # Load individual models
    model_files = {
        "Logistic Regression": models_dir / "logistic_regression.pkl",
        "Random Forest": models_dir / "random_forest.pkl",
        "Support Vector Machine (SVM)": models_dir / "svm_classifier.pkl"
    }
    for name, path in model_files.items():
        if path.exists():
            artifacts["models"][name] = joblib.load(path)
            
    # Load evaluation metrics JSON
    metrics_path = outputs_dir / "model_evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            artifacts["metrics"] = json.load(f)
            
    artifacts["is_ready"] = (len(artifacts["missing_files"]) == 0)
    return artifacts


@st.cache_data
def load_datasets():
    """Loads the accounts and network edge list."""
    accounts_path = Path("data/processed_accounts.csv")
    if not accounts_path.exists():
        accounts_path = Path("data/social_accounts.csv")
    edges_path = Path("data/illustrative_account_edges.csv")
    
    accounts_df = pd.read_csv(accounts_path) if accounts_path.exists() else pd.DataFrame()
    edges_df = pd.read_csv(edges_path) if edges_path.exists() else pd.DataFrame()
    return accounts_df, edges_df


# =====================================================================
# Prediction Helper (Strictly Preserves 32-Feature Formulation)
# =====================================================================
def run_prediction(
    username: str,
    followers: float,
    following: float,
    posts: float,
    account_age_days: float,
    bio: str,
    is_verified: bool,
    graph_context: dict,
    model_name: str,
    artifacts: dict
):
    """
    Constructs a 32-feature vector aligned strictly with training data and runs prediction.
    """
    scaler = artifacts["scaler"]
    vectorizer = artifacts["vectorizer"]
    clf = artifacts["models"].get(model_name, artifacts["best_model"])
    
    # 1. Feature Engineering: Behavioral Ratios
    ff_ratio = followers / (following + 1.0)
    reputation = followers / (followers + following + 1.0)
    ppd = posts / (account_age_days + 1.0)
    ver_val = 1.0 if is_verified else 0.0
    
    # 2. Graph Features (from context or default)
    in_deg = graph_context.get("in_degree", 1.0)
    out_deg = graph_context.get("out_degree", 2.0)
    tot_deg = in_deg + out_deg
    in_out_ratio = in_deg / (out_deg + 1.0)
    pagerank = graph_context.get("pagerank", 0.08)
    betweenness = graph_context.get("betweenness", 0.01)
    clustering = graph_context.get("clustering", 0.3)
    weak_comp = graph_context.get("weak_comp", 10.0)
    strong_comp = graph_context.get("strong_comp", 2.0)
    
    # Align numerical features in exact order of scaler.feature_names_in_
    numerical_dict = {
        "followers": followers,
        "following": following,
        "posts": posts,
        "account_age_days": account_age_days,
        "is_verified": ver_val,
        "follower_following_ratio": ff_ratio,
        "reputation_score": reputation,
        "posts_per_day": ppd,
        "graph_in_degree": in_deg,
        "graph_out_degree": out_deg,
        "graph_total_degree": tot_deg,
        "graph_in_out_ratio": in_out_ratio,
        "graph_pagerank": pagerank,
        "graph_betweenness": betweenness,
        "graph_clustering_coefficient": clustering,
        "graph_weak_component_size": weak_comp,
        "graph_strong_component_size": strong_comp
    }
    
    num_df = pd.DataFrame([numerical_dict])[scaler.feature_names_in_]
    num_scaled = scaler.transform(num_df)
    num_scaled_df = pd.DataFrame(num_scaled, columns=[f"scaled_{c}" for c in scaler.feature_names_in_])
    
    # 3. NLP Feature Extraction using the fitted vectorizer
    clean_handle = clean_username_for_nlp(username)
    clean_bio_text = clean_text(bio)
    combined_text = f"{clean_handle} {clean_bio_text}".strip()
    if not combined_text:
        combined_text = "empty"
        
    tfidf_matrix = vectorizer.transform([combined_text]).toarray()
    tfidf_cols = [f"tfidf_{name.replace(' ', '_')}" for name in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_cols)
    
    # 4. Concatenate to form 32-feature matrix X
    X_sample = pd.concat([num_scaled_df, tfidf_df], axis=1)
    
    # 5. Model Inference
    pred = int(clf.predict(X_sample)[0])
    
    # Confidence estimation
    confidence = 0.5
    prob_fake = 0.5
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_sample)[0]
        prob_fake = float(probs[1])
        confidence = float(probs[pred])
    elif hasattr(clf, "decision_function"):
        df_val = float(clf.decision_function(X_sample)[0])
        prob_fake = float(1.0 / (1.0 + np.exp(-df_val)))
        confidence = prob_fake if pred == 1 else (1.0 - prob_fake)
        
    # Extract detected keyword tokens that contributed to TF-IDF
    detected_keywords = []
    for term, val in zip(vectorizer.get_feature_names_out(), tfidf_matrix[0]):
        if val > 0:
            detected_keywords.append(term)
            
    return {
        "prediction": pred,
        "label_name": "Fake / Suspicious Account" if pred == 1 else "Genuine / Legitimate Account",
        "confidence": confidence,
        "prob_fake": prob_fake,
        "prob_genuine": 1.0 - prob_fake,
        "follower_following_ratio": ff_ratio,
        "reputation_score": reputation,
        "posts_per_day": ppd,
        "detected_keywords": detected_keywords,
        "graph_in_out_ratio": in_out_ratio
    }


# =====================================================================
# Dashboard Page Views
# =====================================================================

def render_home_page(accounts_df, edges_df, artifacts):
    """View 1: Home Command Center Overview"""
    hero_html = (
        '<div style="background:linear-gradient(135deg, rgba(229,9,20,0.15) 0%, rgba(18,18,18,0.95) 100%); '
        'border:1px solid #441414; border-radius:12px; padding:24px 28px; margin-bottom:24px; '
        'box-shadow:0 4px 30px rgba(0,0,0,0.6);">'
        '<div style="display:flex; align-items:center; gap:20px;">'
        f'<div>{CYBERSENTINEL_SHIELD_SVG}</div>'
        '<div>'
        '<h1 style="color:#FFFFFF; font-size:2.2rem; margin:0; font-weight:900; letter-spacing:0.08em;">'
        'CYBERSENTINEL'
        '</h1>'
        '<div style="color:#E50914; font-family:\'Rajdhani\',sans-serif; font-size:1.05rem; font-weight:700; letter-spacing:0.12em;">'
        'INTELLIGENT FAKE ACCOUNT DETECTION SYSTEM'
        '</div>'
        '<div style="color:#B3B3B3; font-size:0.92rem; margin-top:6px;">'
        'Multimodal Cyber Threat Intelligence Framework integrating Profile Analytics, NLP Text TF-IDF, and NetworkX Directed Graph Topology.'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    render_html(hero_html)
    
    # Command Center Metrics Grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        c1_html = (
            '<div class="cyber-card">'
            '<div class="metric-title">⛨ Benchmark Accounts</div>'
            f'<div class="metric-val">{len(accounts_df)}</div>'
            '<span class="metric-badge badge-cyber">Ground Truth Verified</span>'
            '</div>'
        )
        render_html(c1_html)
    with col2:
        genuine_count = sum(accounts_df["label"] == 0) if "label" in accounts_df.columns else 0
        c2_html = (
            '<div class="cyber-card">'
            '<div class="metric-title">🛡️ Genuine Profiles</div>'
            f'<div class="metric-val" style="color:#10B981;">{genuine_count}</div>'
            '<span class="metric-badge badge-genuine">Authentic Identities</span>'
            '</div>'
        )
        render_html(c2_html)
    with col3:
        fake_count = sum(accounts_df["label"] == 1) if "label" in accounts_df.columns else 0
        c3_html = (
            '<div class="cyber-card">'
            '<div class="metric-title">🚨 Fake / Bot Profiles</div>'
            f'<div class="metric-val" style="color:#E50914;">{fake_count}</div>'
            '<span class="metric-badge badge-fake">Sybil Threat Nodes</span>'
            '</div>'
        )
        render_html(c3_html)
    with col4:
        c4_html = (
            '<div class="cyber-card">'
            '<div class="metric-title">⛯ Follower Network Edges</div>'
            f'<div class="metric-val">{len(edges_df)}</div>'
            '<span class="metric-badge badge-cyber">Directed Interaction Links</span>'
            '</div>'
        )
        render_html(c4_html)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⛯ TECHNICAL ARCHITECTURE PIPELINE")
    
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        pipe1_html = (
            '<div class="cyber-card" style="min-height:235px;">'
            '<div style="color:#E50914; font-family:\'Orbitron\',monospace; font-size:0.85rem; font-weight:700;">STAGE 01</div>'
            '<h4 style="color:#FFFFFF; margin:4px 0 10px 0;">Data Preprocessing</h4>'
            '<p style="font-size:0.86rem; color:#B3B3B3; line-height:1.5;">'
            '• Exact duplicate and username integrity audits.<br>'
            '• Imputation of missing profile attributes.<br>'
            '• Strict boolean type standardization.<br>'
            '• Preserves immutable raw historical baselines.'
            '</p>'
            '</div>'
        )
        render_html(pipe1_html)
    with p2:
        pipe2_html = (
            '<div class="cyber-card" style="min-height:235px;">'
            '<div style="color:#E50914; font-family:\'Orbitron\',monospace; font-size:0.85rem; font-weight:700;">STAGE 02</div>'
            '<h4 style="color:#FFFFFF; margin:4px 0 10px 0;">Multimodal Features</h4>'
            '<p style="font-size:0.86rem; color:#B3B3B3; line-height:1.5;">'
            '• <b>Profile (8)</b>: Follower ratios, reputation, post rates.<br>'
            '• <b>Text (15)</b>: Combined handle & bio TF-IDF n-grams.<br>'
            '• <b>Graph (9)</b>: PageRank, in/out degrees, clustering, SCC rings.'
            '</p>'
            '</div>'
        )
        render_html(pipe2_html)
    with p3:
        pipe3_html = (
            '<div class="cyber-card" style="min-height:235px;">'
            '<div style="color:#E50914; font-family:\'Orbitron\',monospace; font-size:0.85rem; font-weight:700;">STAGE 03</div>'
            '<h4 style="color:#FFFFFF; margin:4px 0 10px 0;">Zero-Leakage ML</h4>'
            '<p style="font-size:0.86rem; color:#B3B3B3; line-height:1.5;">'
            '• Stratified Train/Test partition isolation.<br>'
            '• StandardScaler & TF-IDF fitted <i>only</i> on Train partition.<br>'
            '• Ground-truth labels isolated strictly from feature space.<br>'
            '• Models: Logistic Regression, Random Forest, SVM.'
            '</p>'
            '</div>'
        )
        render_html(pipe3_html)
    with p4:
        pipe4_html = (
            '<div class="cyber-card" style="min-height:235px;">'
            '<div style="color:#E50914; font-family:\'Orbitron\',monospace; font-size:0.85rem; font-weight:700;">STAGE 04</div>'
            '<h4 style="color:#FFFFFF; margin:4px 0 10px 0;">Real-Time Defense</h4>'
            '<p style="font-size:0.86rem; color:#B3B3B3; line-height:1.5;">'
            '• Real-time behavioral inference & confidence scoring.<br>'
            '• Interactive Plotly threat intelligence graph viewer.<br>'
            '• Comparative model diagnostics & confusion matrices.<br>'
            '• Deep dataset investigation & attribute filtering.'
            '</p>'
            '</div>'
        )
        render_html(pipe4_html)
        
    notice_html = (
        '<div class="info-pill">'
        '<b>🛡️ Academic Prototype Notice</b>: The current evaluation dataset incorporates an illustrative 10-account baseline '
        'calibrated for architectural validation, algorithmic verification, and defense presentation demonstrations. '
        'Production deployments can connect to large-scale benchmark threat repositories such as <b>Cresci-2015/2017</b> or <b>TwiBot-20/22</b>.'
        '</div>'
    )
    render_html(notice_html)


def render_prediction_page(artifacts, accounts_df, edges_df):
    """View 2: Account Prediction Form & Reasoning"""
    page_header_html = (
        '<div style="margin-bottom:20px;">'
        '<h2 style="margin:0; color:#FFFFFF;">🎯 REAL-TIME ACCOUNT THREAT PREDICTION</h2>'
        '<p style="color:#B3B3B3; margin-top:4px;">'
        'Evaluate profile attributes, linguistic bio text tokens, and structural graph topology to detect malicious bot behavior.'
        '</p>'
        '</div>'
    )
    render_html(page_header_html)
    
    if not artifacts["is_ready"]:
        st.error(f"Models are not ready. Missing files: {artifacts['missing_files']}. Run model training first.")
        return
        
    st.markdown("#### ⚡ QUICK PRESET BENCHMARK PROFILES")
    sample_usernames = list(accounts_df["username"]) if not accounts_df.empty else []
    selected_preset = st.selectbox(
        "Select a verified benchmark profile to auto-populate fields, or choose 'Custom Input':",
        ["Custom Input"] + sample_usernames
    )
    
    preset_data = {}
    if selected_preset != "Custom Input" and not accounts_df.empty:
        row = accounts_df[accounts_df["username"] == selected_preset].iloc[0]
        preset_data = {
            "username": str(row["username"]),
            "followers": int(row["followers"]),
            "following": int(row["following"]),
            "posts": int(row["posts"]),
            "account_age_days": int(row["account_age_days"]),
            "bio": str(row["bio"]),
            "is_verified": bool(row["is_verified"])
        }
    else:
        preset_data = {
            "username": "crypto_promo_99",
            "followers": 12,
            "following": 4500,
            "posts": 1,
            "account_age_days": 4,
            "bio": "Make money fast click my link win prizes",
            "is_verified": False
        }
        
    with st.form("account_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            h1_html = (
                '<div style="border-bottom:1px solid #441414; padding-bottom:6px; margin-bottom:14px;">'
                '<h5 style="color:#E50914; margin:0; letter-spacing:0.05em;">👤 PROFILE IDENTIFIERS & VOLUME</h5>'
                '</div>'
            )
            render_html(h1_html)
            u_name = st.text_input("Account Username / Handle", value=preset_data["username"])
            followers = st.number_input("Followers Count", min_value=0, max_value=100000000, value=preset_data["followers"], step=10)
            following = st.number_input("Following Count", min_value=0, max_value=100000000, value=preset_data["following"], step=10)
            is_verified = st.checkbox("Profile Possesses Verified Badge?", value=preset_data["is_verified"])
            
        with col2:
            h2_html = (
                '<div style="border-bottom:1px solid #441414; padding-bottom:6px; margin-bottom:14px;">'
                '<h5 style="color:#E50914; margin:0; letter-spacing:0.05em;">📝 ACTIVITY HISTORY & BIOGRAPHY</h5>'
                '</div>'
            )
            render_html(h2_html)
            posts = st.number_input("Total Published Posts", min_value=0, max_value=1000000, value=preset_data["posts"], step=5)
            age = st.number_input("Account Age (Days)", min_value=0, max_value=15000, value=preset_data["account_age_days"], step=10)
            bio = st.text_area("Profile Biography Text (Bio)", value=preset_data["bio"], height=95)
            
        h3_html = (
            '<div style="border-bottom:1px solid #441414; padding-bottom:6px; margin:18px 0 14px 0;">'
            '<h5 style="color:#E50914; margin:0; letter-spacing:0.05em;">⛯ NETWORK GRAPH TOPOLOGY CONTEXT</h5>'
            '</div>'
        )
        render_html(h3_html)
        
        c_net1, c_net2 = st.columns(2)
        with c_net1:
            graph_mode = st.radio(
                "Graph Context Source",
                ["Match with existing network graph", "Simulate as new account (custom incoming/outgoing follows)"],
                index=0
            )
        with c_net2:
            sim_in_deg = st.slider("Simulated Incoming Follows", 0, 20, 1) if "Simulate" in graph_mode else 1
            sim_out_deg = st.slider("Simulated Outgoing Follows", 0, 20, 4) if "Simulate" in graph_mode else 3
            
        model_choice = st.selectbox(
            "Classifier Model to Deploy for Inference:",
            list(artifacts["models"].keys()),
            index=0
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🔍 RUN THREAT INFERENCE ANALYSIS", use_container_width=True)
        
    if submit_btn:
        G = build_account_graph(list(accounts_df["username"]), edge_list_path="data/illustrative_account_edges.csv")
        graph_ctx = {}
        
        if "Match" in graph_mode and u_name in G:
            in_deg = float(G.in_degree(u_name))
            out_deg = float(G.out_degree(u_name))
            pr_dict = nx.pagerank(G, alpha=0.85)
            bw_dict = nx.betweenness_centrality(G)
            cl_dict = nx.clustering(G.to_undirected())
            graph_ctx = {
                "in_degree": in_deg,
                "out_degree": out_deg,
                "pagerank": float(pr_dict.get(u_name, 0.08)),
                "betweenness": float(bw_dict.get(u_name, 0.01)),
                "clustering": float(cl_dict.get(u_name, 0.3)),
                "weak_comp": 10.0,
                "strong_comp": 2.0
            }
        else:
            graph_ctx = {
                "in_degree": float(sim_in_deg),
                "out_degree": float(sim_out_deg),
                "pagerank": 0.05 if sim_in_deg < 2 else 0.15,
                "betweenness": 0.01,
                "clustering": 0.2,
                "weak_comp": 10.0,
                "strong_comp": 1.0 if sim_in_deg == 0 else 2.0
            }
            
        res = run_prediction(
            username=u_name,
            followers=followers,
            following=following,
            posts=posts,
            account_age_days=age,
            bio=bio,
            is_verified=is_verified,
            graph_context=graph_ctx,
            model_name=model_choice,
            artifacts=artifacts
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 PREDICTION RESULTS & EXPLAINABILITY AUDIT")
        
        p_col1, p_col2 = st.columns([1.1, 1.3])
        
        with p_col1:
            if res["prediction"] == 1:
                alert_html = (
                    '<div class="alert-card-fake">'
                    '<div style="font-family:\'Orbitron\',monospace; font-size:1.25rem; font-weight:800; color:#E50914; margin-bottom:6px;">'
                    '🚨 CRITICAL THREAT: FAKE / BOT ACCOUNT'
                    '</div>'
                    '<div style="color:#FFFFFF; font-size:1.05rem;">'
                    f'Model Confidence: <b style="color:#E50914;">{res["confidence"]*100:.1f}%</b>'
                    '</div>'
                    '<div style="color:#FFA3A3; font-size:0.85rem; margin-top:6px;">'
                    f'Classified as Sybil Spambot by <b>{model_choice}</b> based on multimodal behavioral fingerprint.'
                    '</div>'
                    '</div>'
                )
                render_html(alert_html)
            else:
                alert_html = (
                    '<div class="alert-card-genuine">'
                    '<div style="font-family:\'Orbitron\',monospace; font-size:1.25rem; font-weight:800; color:#10B981; margin-bottom:6px;">'
                    '🛡️ VERIFIED AUTHENTIC: GENUINE ACCOUNT'
                    '</div>'
                    '<div style="color:#FFFFFF; font-size:1.05rem;">'
                    f'Model Confidence: <b style="color:#10B981;">{res["confidence"]*100:.1f}%</b>'
                    '</div>'
                    '<div style="color:#A7F3D0; font-size:0.85rem; margin-top:6px;">'
                    f'Classified as Legitimate User by <b>{model_choice}</b> with normal interaction topology.'
                    '</div>'
                    '</div>'
                )
                render_html(alert_html)
                
            fig_prob = go.Figure(go.Bar(
                x=[res["prob_genuine"], res["prob_fake"]],
                y=["Genuine", "Fake"],
                orientation="h",
                marker=dict(
                    color=["#10B981", "#E50914"],
                    line=dict(color=["#054E3B", "#8B0000"], width=1.5)
                ),
                text=[f"{res['prob_genuine']*100:.1f}%", f"{res['prob_fake']*100:.1f}%"],
                textposition="auto"
            ))
            fig_prob.update_layout(
                title=dict(text="Class Probability Distribution", font=dict(family="Orbitron", color="#FFFFFF", size=14)),
                xaxis_title="Confidence Probability",
                xaxis=dict(range=[0, 1.05], gridcolor="#222222", zerolinecolor="#441414"),
                yaxis=dict(gridcolor="#222222"),
                height=220,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(18,18,18,0.7)",
                font=dict(color="#B3B3B3")
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with p_col2:
            ratio_status = "<span style='color:#E50914; font-weight:600;'>⚠️ Highly Suspicious (&lt; 0.05)</span>" if res["follower_following_ratio"] < 0.05 else "<span style='color:#10B981; font-weight:600;'>✓ Balanced</span>"
            if res["detected_keywords"]:
                keywords_str = " ".join([f"<span class='metric-badge badge-fake'>{k}</span>" for k in res["detected_keywords"]])
            else:
                keywords_str = "<span class='metric-badge badge-genuine'>Clean / Neutral</span>"
                
            signals_html = (
                '<div class="cyber-card">'
                '<h4 style="color:#FFFFFF; margin-top:0; margin-bottom:12px; border-bottom:1px solid #441414; padding-bottom:8px;">'
                '🧠 Influential Decision Signals'
                '</h4>'
                f'• <b>Follower / Following Ratio</b>: <code>{res["follower_following_ratio"]:.4f}</code> — {ratio_status}<br><br>'
                f'• <b>Reputation Score</b>: <code>{res["reputation_score"]:.4f}</code> <i style="color:#888;">(Standard Baseline: 0.50–1.00)</i><br><br>'
                f'• <b>Activity Intensity</b>: <code>{res["posts_per_day"]:.2f}</code> posts/day<br><br>'
                f'• <b>Linguistic Spam Tokens</b>: {keywords_str}<br><br>'
                f'• <b>Network Graph In/Out Ratio</b>: <code>{res["graph_in_out_ratio"]:.2f}</code>'
                '</div>'
            )
            render_html(signals_html)


def render_graph_analysis_page(accounts_df, edges_df):
    """View 3: Threat Intelligence Network Topology Analysis"""
    header_html = (
        '<div style="margin-bottom:15px;">'
        '<h2 style="margin:0; color:#FFFFFF;">⛯ NETWORK THREAT TOPOLOGY INVESTIGATION</h2>'
        '<p style="color:#B3B3B3; margin-top:4px;">'
        'Interactive cyber threat intelligence graph constructed with <b>NetworkX</b> and rendered with <b>Plotly</b>.<br>'
        'Directed edges delineate follower relationships (<code>source</code> → <code>target</code>).'
        '</p>'
        '</div>'
    )
    render_html(header_html)
    
    warn_html = (
        '<div class="info-pill" style="border-left-color:#8B0000;">'
        '⚠️ <b>Network Topology Calibration</b>: Directed connections visualize the baseline benchmark edge list '
        '(<code>data/illustrative_account_edges.csv</code>). Designed to highlight centrality anomalies, in-degree clustering, '
        'and sybil bot rings.'
        '</div>'
    )
    render_html(warn_html)
    
    if edges_df.empty or accounts_df.empty:
        st.error("Graph edges or accounts data file missing.")
        return
        
    usernames = list(accounts_df["username"])
    G = build_account_graph(usernames, edge_list_path="data/illustrative_account_edges.csv")
    
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    pageranks = nx.pagerank(G, alpha=0.85)
    clustering = nx.clustering(G.to_undirected())
    labels_map = dict(zip(accounts_df["username"], accounts_df["label"]))
    pos = nx.spring_layout(G, seed=42, k=0.9)
    
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="rgba(229, 9, 20, 0.35)"),
        hoverinfo="none",
        mode="lines"
    )
    
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        is_fake = labels_map.get(node, 0) == 1
        color = "#E50914" if is_fake else "#10B981"
        node_colors.append(color)
        
        size = 20 + in_degrees.get(node, 0) * 5
        node_sizes.append(size)
        
        node_info = (
            f"<b>@{node}</b><br>"
            f"Status: {'THREAT / FAKE' if is_fake else 'AUTHENTIC / GENUINE'}<br>"
            f"In-Degree (Followers): {in_degrees.get(node, 0)}<br>"
            f"Out-Degree (Following): {out_degrees.get(node, 0)}<br>"
            f"PageRank Score: {pageranks.get(node, 0.0):.4f}<br>"
            f"Clustering Coeff: {clustering.get(node, 0.0):.4f}"
        )
        node_text.append(node_info)
        
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2.5, color="#FFFFFF")
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Social Relationship Interaction Graph (Green = Genuine, Crimson Red = Fake/Bot)",
                font=dict(family="Orbitron", color="#FFFFFF", size=15)
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=55),
            paper_bgcolor="#0c0c0c",
            plot_bgcolor="#121212",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔬 DEEP NODE METRIC INSPECTION")
    selected_node = st.selectbox(
        "Select an account node to inspect detailed topological metrics:", 
        list(G.nodes())
    )
    
    if selected_node:
        is_fake = labels_map.get(selected_node, 0) == 1
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        
        with col_m1:
            m1_html = (
                '<div class="cyber-card">'
                '<div class="metric-title">Classification</div>'
                f'<div class="metric-val" style="font-size:1.3rem; color:{"#E50914" if is_fake else "#10B981"};">'
                f'{"Fake (1)" if is_fake else "Genuine (0)"}'
                '</div>'
                '</div>'
            )
            render_html(m1_html)
        with col_m2:
            m2_html = (
                '<div class="cyber-card">'
                '<div class="metric-title">In-Degree</div>'
                f'<div class="metric-val">{in_degrees.get(selected_node, 0)}</div>'
                '</div>'
            )
            render_html(m2_html)
        with col_m3:
            m3_html = (
                '<div class="cyber-card">'
                '<div class="metric-title">Out-Degree</div>'
                f'<div class="metric-val">{out_degrees.get(selected_node, 0)}</div>'
                '</div>'
            )
            render_html(m3_html)
        with col_m4:
            m4_html = (
                '<div class="cyber-card">'
                '<div class="metric-title">PageRank</div>'
                f'<div class="metric-val" style="font-size:1.4rem;">{pageranks.get(selected_node, 0):.4f}</div>'
                '</div>'
            )
            render_html(m4_html)
        with col_m5:
            m5_html = (
                '<div class="cyber-card">'
                '<div class="metric-title">Clustering</div>'
                f'<div class="metric-val" style="font-size:1.4rem;">{clustering.get(selected_node, 0):.4f}</div>'
                '</div>'
            )
            render_html(m5_html)


def render_model_performance_page(artifacts):
    """View 4: Model Evaluation and Comparison"""
    header_html = (
        '<div style="margin-bottom:20px;">'
        '<h2 style="margin:0; color:#FFFFFF;">📈 MODEL PERFORMANCE & COMPARATIVE BENCHMARK</h2>'
        '<p style="color:#B3B3B3; margin-top:4px;">'
        'Empirical evaluation metrics computed on the isolated test partition during Phase 3 model training.'
        '</p>'
        '</div>'
    )
    render_html(header_html)
    
    metrics_data = artifacts["metrics"]
    if not metrics_data:
        st.warning("Saved evaluation metrics not found in `outputs/model_evaluation_metrics.json`. Run model training first.")
        return
        
    results_summary = metrics_data["results_summary"]
    best_model = metrics_data.get("best_model", "Logistic Regression")
    
    banner_html = (
        '<div style="background:linear-gradient(135deg, rgba(229,9,20,0.25) 0%, rgba(18,18,18,0.95) 100%); '
        'border:1px solid #E50914; border-radius:8px; padding:18px 24px; margin-bottom:22px; '
        'display:flex; align-items:center; justify-content:space-between;">'
        '<div>'
        '<span class="metric-badge badge-fake" style="margin:0 0 6px 0;">CHAMPION MODEL</span>'
        f'<div style="font-family:\'Orbitron\',monospace; font-size:1.5rem; color:#FFFFFF; font-weight:800;">{best_model}</div>'
        '<div style="color:#B3B3B3; font-size:0.9rem; margin-top:4px;">'
        'Selected as optimal production classifier based on balanced F1-score and architectural generalization simplicity.'
        '</div>'
        '</div>'
        '<div style="font-size:2.5rem; color:#E50914;">🏆</div>'
        '</div>'
    )
    render_html(banner_html)
    
    models = list(results_summary.keys())
    perf_rows = []
    for m in models:
        r = results_summary[m]
        perf_rows.append({
            "Classifier Model": m,
            "Accuracy": f"{r['accuracy']:.2f}",
            "Precision (Fake)": f"{r['precision']:.2f}",
            "Recall (Fake)": f"{r['recall']:.2f}",
            "F1-Score (Fake)": f"{r['f1_score']:.2f}",
            "Train Accuracy": f"{r['train_accuracy']:.2f}"
        })
    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 MULTI-METRIC PERFORMANCE MATRIX")
    plot_data = []
    for m in models:
        r = results_summary[m]
        plot_data.append({"Model": m, "Metric": "Accuracy", "Score": r["accuracy"]})
        plot_data.append({"Model": m, "Metric": "Precision", "Score": r["precision"]})
        plot_data.append({"Model": m, "Metric": "Recall", "Score": r["recall"]})
        plot_data.append({"Model": m, "Metric": "F1-Score", "Score": r["f1_score"]})
        
    bar_df = pd.DataFrame(plot_data)
    cyber_palette = ["#E50914", "#8B0000", "#B3B3B3"]
    fig_bar = px.bar(
        bar_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=cyber_palette,
        title="Comparative Classifier Scores on Isolated Test Set"
    )
    fig_bar.update_layout(
        yaxis=dict(range=[0, 1.15], gridcolor="#222222"),
        xaxis=dict(gridcolor="#222222"),
        paper_bgcolor="#0c0c0c",
        plot_bgcolor="#141414",
        font=dict(family="Rajdhani", color="#FFFFFF", size=13),
        title_font=dict(family="Orbitron", color="#FFFFFF", size=14)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔲 CONFUSION MATRICES (TEST PARTITION)")
    cm_cols = st.columns(len(models))
    for idx, m in enumerate(models):
        cm = np.array(results_summary[m]["confusion_matrix"])
        with cm_cols[idx]:
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted Class", y="True Ground Truth"),
                x=["Genuine (0)", "Fake (1)"],
                y=["Genuine (0)", "Fake (1)"],
                color_continuous_scale="Reds",
                title=f"{m}"
            )
            fig_cm.update_layout(
                height=320,
                paper_bgcolor="#0c0c0c",
                plot_bgcolor="#141414",
                font=dict(family="Rajdhani", color="#FFFFFF"),
                title_font=dict(family="Orbitron", color="#FFFFFF", size=13)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
    note_html = (
        '<div class="info-pill">'
        '<b>🛡️ Academic Research Note</b>: The 1.00 (100%) metric scores observed here reflect the small illustrative sample size '
        '(N=10; 7 train, 3 test) where patterns are cleanly separated. In your B.Tech thesis, present these as pipeline verification '
        'benchmarks and recommend validating on public corpora like <b>Cresci-2015/2017</b> or <b>TwiBot-20/22</b>.'
        '</div>'
    )
    render_html(note_html)


def render_dataset_explorer_page(accounts_df):
    """View 5: Dataset Exploration and Interactive Filters"""
    header_html = (
        '<div style="margin-bottom:20px;">'
        '<h2 style="margin:0; color:#FFFFFF;">📁 DATASET THREAT EXPLORATION & AUDIT</h2>'
        '<p style="color:#B3B3B3; margin-top:4px;">'
        'Inspect, search, and audit preprocessed benchmark records with interactive attribute filters.'
        '</p>'
        '</div>'
    )
    render_html(header_html)
    
    if accounts_df.empty:
        st.error("Accounts dataset not found.")
        return
        
    c_f1, c_f2, c_f3 = st.columns([1.5, 1, 1])
    with c_f1:
        search_query = st.text_input("🔍 Search by Username or Bio Keyword:", value="")
    with c_f2:
        class_filter = st.selectbox("Filter by Class Label:", ["All Accounts", "Genuine Accounts (0)", "Fake Accounts (1)"])
    with c_f3:
        verified_filter = st.selectbox("Filter by Verification Badge:", ["All", "Verified Only", "Unverified Only"])
        
    filtered_df = accounts_df.copy()
    if search_query:
        mask = (
            filtered_df["username"].astype(str).str.contains(search_query, case=False, na=False) |
            filtered_df["bio"].astype(str).str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
        
    if "Genuine" in class_filter:
        filtered_df = filtered_df[filtered_df["label"] == 0]
    elif "Fake" in class_filter:
        filtered_df = filtered_df[filtered_df["label"] == 1]
        
    if "Verified Only" in verified_filter:
        filtered_df = filtered_df[filtered_df["is_verified"].astype(bool) == True]
    elif "Unverified Only" in verified_filter:
        filtered_df = filtered_df[filtered_df["is_verified"].astype(bool) == False]
        
    st.markdown(f"**Showing `{len(filtered_df)}` of `{len(accounts_df)}` benchmark accounts:**")
    st.dataframe(filtered_df, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 DATASET DISTRIBUTION INSIGHTS")
    c_d1, c_d2 = st.columns(2)
    
    with c_d1:
        label_counts = accounts_df["label"].value_counts().reset_index()
        label_counts["Class"] = label_counts["label"].map({0: "Genuine (0)", 1: "Fake (1)"})
        fig_donut = px.pie(
            label_counts,
            values="count",
            names="Class",
            hole=0.55,
            color="Class",
            color_discrete_map={"Genuine (0)": "#10B981", "Fake (1)": "#E50914"},
            title="Class Distribution Balance"
        )
        fig_donut.update_layout(
            paper_bgcolor="#0c0c0c",
            plot_bgcolor="#141414",
            font=dict(family="Rajdhani", color="#FFFFFF", size=13),
            title_font=dict(family="Orbitron", color="#FFFFFF", size=14)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with c_d2:
        fig_scatter = px.scatter(
            accounts_df,
            x="following",
            y="followers",
            color=accounts_df["label"].map({0: "Genuine", 1: "Fake"}),
            color_discrete_map={"Genuine": "#10B981", "Fake": "#E50914"},
            hover_name="username",
            size="posts",
            log_x=True,
            log_y=True,
            title="Followers vs Following (Log Scale, Node Size = Posts)"
        )
        fig_scatter.update_layout(
            xaxis=dict(gridcolor="#222222"),
            yaxis=dict(gridcolor="#222222"),
            paper_bgcolor="#0c0c0c",
            plot_bgcolor="#141414",
            font=dict(family="Rajdhani", color="#FFFFFF", size=13),
            title_font=dict(family="Orbitron", color="#FFFFFF", size=14)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# =====================================================================
# Sidebar Branding & Status Rendering (Clean DOM Rendering)
# =====================================================================
def render_sidebar_header():
    """Renders the top branding header in the sidebar without markdown code block leakage."""
    branding_html = (
        '<div style="text-align:center; padding:10px 0 6px 0;">'
        '<div style="display:flex; justify-content:center; align-items:center; margin-bottom:10px;">'
        f'{CYBERSENTINEL_SHIELD_SVG}'
        '</div>'
        '<div style="font-family:\'Orbitron\',\'Rajdhani\',sans-serif; font-size:1.45rem; font-weight:900; letter-spacing:0.12em; color:#FFFFFF; line-height:1.2; text-shadow:0 0 10px rgba(229,9,20,0.4); margin:0;">'
        'CYBERSENTINEL'
        '</div>'
        '<div style="font-family:\'Rajdhani\',\'Inter\',sans-serif; font-size:0.72rem; font-weight:700; color:#B3B3B3; letter-spacing:0.1em; margin-top:5px; text-transform:uppercase;">'
        'INTELLIGENT FAKE ACCOUNT DETECTION SYSTEM'
        '</div>'
        '</div>'
        '<hr style="border:none; height:1px; background:linear-gradient(90deg, transparent, #E50914, #441414, transparent); margin:14px 0 16px 0;">'
    )
    render_html(branding_html, in_sidebar=True)


def render_sidebar_status(artifacts):
    """Renders the defense engine status card in the sidebar."""
    divider_html = (
        '<hr style="border:none; height:1px; background:linear-gradient(90deg, transparent, #441414, #E50914, transparent); margin:18px 0 14px 0;">'
        '<div style="font-family:\'Orbitron\',sans-serif; font-size:0.85rem; font-weight:700; color:#FFFFFF; letter-spacing:0.08em; margin-bottom:10px;">'
        '📌 SYSTEM STATUS'
        '</div>'
    )
    render_html(divider_html, in_sidebar=True)

    if artifacts["is_ready"]:
        best_mod = artifacts["metrics"].get("best_model", "Logistic Regression") if artifacts["metrics"] else "Logistic Regression"
        status_html = (
            '<div style="background:#141414; border:1px solid #441414; border-radius:6px; padding:12px; margin-bottom:12px;">'
            '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
            '<span style="font-size:0.75rem; color:#B3B3B3; text-transform:uppercase; font-weight:600;">DEFENSE ENGINE</span>'
            '<span style="color:#10B981; font-family:\'Orbitron\',monospace; font-weight:700; font-size:0.8rem; background:rgba(16,185,129,0.15); padding:1px 6px; border-radius:3px; border:1px solid rgba(16,185,129,0.3);">● ACTIVE</span>'
            '</div>'
            '<div style="font-size:0.72rem; color:#888888; text-transform:uppercase;">CHAMPION MODEL</div>'
            f'<div style="color:#E50914; font-family:\'Rajdhani\',sans-serif; font-weight:700; font-size:0.95rem; margin-bottom:6px;">{best_mod}</div>'
            '<div style="font-size:0.72rem; color:#888888; text-transform:uppercase;">PIPELINE INTEGRATION</div>'
            '<div style="color:#FFFFFF; font-size:0.82rem;">Profile + TF-IDF + Graph</div>'
            '</div>'
        )
        render_html(status_html, in_sidebar=True)
    else:
        st.sidebar.error("❌ Models Missing. Run model training.")

    footer_html = (
        '<div style="font-family:\'Rajdhani\',sans-serif; font-size:0.75rem; color:#666666; text-align:center; margin-top:24px; padding-top:12px; border-top:1px solid #222222;">'
        '<b style="color:#888888;">B.Tech Final-Year Capstone Project</b><br>'
        'Domain: Generative AI & Cybersecurity<br>'
        'Streamlit • Scikit-Learn • NetworkX'
        '</div>'
    )
    render_html(footer_html, in_sidebar=True)


# =====================================================================
# Main Application Entry Point
# =====================================================================
def main():
    apply_cybersecurity_theme()
    
    # Load resources
    artifacts = load_models_and_artifacts()
    accounts_df, edges_df = load_datasets()
    
    # Render Sidebar Branding
    render_sidebar_header()
    
    # Professional Sidebar Navigation
    nav_selection = st.sidebar.radio(
        "NAVIGATION COMMANDS",
        [
            "⛨ Home Overview",
            "🎯 Account Prediction",
            "⛯ Network Graph Analysis",
            "📈 Model Performance",
            "📁 Dataset Explorer"
        ],
        index=0
    )
    
    # Render Sidebar System Status
    render_sidebar_status(artifacts)
    
    # Page Router
    if "Home Overview" in nav_selection:
        render_home_page(accounts_df, edges_df, artifacts)
    elif "Account Prediction" in nav_selection:
        render_prediction_page(artifacts, accounts_df, edges_df)
    elif "Network Graph Analysis" in nav_selection:
        render_graph_analysis_page(accounts_df, edges_df)
    elif "Model Performance" in nav_selection:
        render_model_performance_page(artifacts)
    elif "Dataset Explorer" in nav_selection:
        render_dataset_explorer_page(accounts_df)


if __name__ == "__main__":
    main()
