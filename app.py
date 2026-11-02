"""
app.py - Streamlit Cybersecurity Dashboard for Fake Social-Media Account Detection.

Project: "A Graph-and-Text Feature Classifier-Based Approach to Fake Social-Media Account Detection"
Academic Note:
    This dashboard serves as an interactive decision-support and demonstration interface
    for testing multimodal (Profile, NLP Text, Network Topology) classification pipelines.
    The benchmark dataset is an illustrative prototype for engineering validation.
"""

import os
import sys
import json
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
# Page Configuration & Cybersecurity Theming
# =====================================================================
st.set_page_config(
    page_title="CyberSentinel | Fake Social Account Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_cybersecurity_theme():
    """Injects dark-navy and cyber-blue CSS for a polished security-operations UI."""
    st.markdown(
        """
        <style>
        /* Base typography and clean backgrounds */
        .stApp {
            background: radial-gradient(circle at 10% 20%, #0d1527 0%, #070b14 90%);
            color: #e2e8f0;
            font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
        }
        
        /* Metric Cards */
        .cyber-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.75), rgba(15, 23, 42, 0.85));
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
            margin-bottom: 16px;
        }
        
        .metric-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin-bottom: 4px;
        }
        
        .metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #38bdf8;
            margin: 0;
        }
        
        .metric-badge {
            display: inline-block;
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
            margin-top: 4px;
        }
        
        .badge-genuine {
            background-color: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }
        
        .badge-fake {
            background-color: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }
        
        /* Alert banners */
        .info-pill {
            background: rgba(14, 165, 233, 0.12);
            border-left: 4px solid #0284c7;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 0.9rem;
            margin: 12px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


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
# Prediction Helper
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

def render_home_page(accounts_df, edges_df):
    """View 1: Home and Overview"""
    st.markdown(
        """
        <div style="text-align: center; padding: 25px 0 10px 0;">
            <h1 style="color: #38bdf8; font-size: 2.3rem; margin-bottom: 6px;">
                🛡️ CyberSentinel: Fake Social-Media Account Detection
            </h1>
            <p style="font-size: 1.1rem; color: #94a3b8; max-width: 850px; margin: 0 auto;">
                A Multimodal Classification Framework Integrating <b>Profile Metrics</b>, 
                <b>NLP Text TF-IDF</b>, and <b>NetworkX Directed Graph Topology</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="cyber-card">
                <div class="metric-title">Total Benchmark Accounts</div>
                <div class="metric-val">{len(accounts_df)}</div>
                <span class="metric-badge badge-genuine">Prototype Benchmark</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        genuine_count = sum(accounts_df["label"] == 0) if "label" in accounts_df.columns else 0
        st.markdown(
            f"""
            <div class="cyber-card">
                <div class="metric-title">Genuine Profiles</div>
                <div class="metric-val">{genuine_count}</div>
                <span class="metric-badge badge-genuine">Verified / Authentic</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        fake_count = sum(accounts_df["label"] == 1) if "label" in accounts_df.columns else 0
        st.markdown(
            f"""
            <div class="cyber-card">
                <div class="metric-title">Fake / Bot Profiles</div>
                <div class="metric-val">{fake_count}</div>
                <span class="metric-badge badge-fake">Spambots / Clones</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="cyber-card">
                <div class="metric-title">Graph Relationship Edges</div>
                <div class="metric-val">{len(edges_df)}</div>
                <span class="metric-badge badge-genuine">Follow Directed Edges</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("### 🔍 Technical Architecture Pipeline")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="cyber-card" style="height: 220px;">
                <h4 style="color: #38bdf8;">1. Data Preprocessing</h4>
                <p style="font-size: 0.85rem; color: #cbd5e1;">
                    • Exact duplicate and username checks.<br>
                    • Imputation of missing values.<br>
                    • Boolean standardization.<br>
                    • Preserves raw data intact.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="cyber-card" style="height: 220px;">
                <h4 style="color: #38bdf8;">2. Multimodal Features</h4>
                <p style="font-size: 0.85rem; color: #cbd5e1;">
                    • <b>Profile</b>: Follower ratios, age, post rate.<br>
                    • <b>Text</b>: Combined handle & bio TF-IDF.<br>
                    • <b>Graph</b>: In/Out degrees, PageRank, clustering, SCC rings.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            """
            <div class="cyber-card" style="height: 220px;">
                <h4 style="color: #38bdf8;">3. Zero-Leakage ML</h4>
                <p style="font-size: 0.85rem; color: #cbd5e1;">
                    • Stratified Train/Test isolation.<br>
                    • Scaler & TF-IDF fitted <i>only</i> on Train.<br>
                    • Label strictly isolated from features.<br>
                    • Models: Logistic Regression, RF, SVM.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            """
            <div class="cyber-card" style="height: 220px;">
                <h4 style="color: #38bdf8;">4. Interactive Defense</h4>
                <p style="font-size: 0.85rem; color: #cbd5e1;">
                    • Real-time account inference.<br>
                    • Plotly interactive network graph.<br>
                    • Performance comparison reports.<br>
                    • Complete dataset exploration.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.info(
        "📌 **Academic Prototype Notice**: The current dataset contains an illustrative 10-account prototype "
        "designed for validating the multimodal pipeline architecture and demonstrating defense presentations. "
        "Real-world deployments require large-scale benchmark datasets such as Cresci-2015/2017 or TwiBot-20/22."
    )


def render_prediction_page(artifacts, accounts_df, edges_df):
    """View 2: Account Prediction Form & Reasoning"""
    st.markdown("## 🎯 Real-Time Account Prediction")
    st.markdown("Evaluate an account's profile attributes, bio text, and network context to detect fake or bot behavior.")
    
    if not artifacts["is_ready"]:
        st.error(f"Models are not ready. Missing files: {artifacts['missing_files']}. Run `python src/model_training.py` first.")
        return
        
    # Pre-population helper
    st.markdown("#### ⚡ Quick Preset Test Accounts")
    sample_usernames = list(accounts_df["username"]) if not accounts_df.empty else []
    selected_preset = st.selectbox(
        "Select an illustrative sample account to pre-fill the form, or choose 'Custom Input':",
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
        
    # Input Form
    with st.form("account_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 👤 Profile Identifiers & Stats")
            u_name = st.text_input("Username / Account Handle", value=preset_data["username"])
            followers = st.number_input("Followers Count", min_value=0, max_value=100000000, value=preset_data["followers"], step=10)
            following = st.number_input("Following Count", min_value=0, max_value=100000000, value=preset_data["following"], step=10)
            is_verified = st.checkbox("Profile Has Blue Verification Badge?", value=preset_data["is_verified"])
            
        with col2:
            st.markdown("##### 📝 Activity & Content")
            posts = st.number_input("Total Published Posts", min_value=0, max_value=1000000, value=preset_data["posts"], step=5)
            age = st.number_input("Account Age (Days)", min_value=0, max_value=15000, value=preset_data["account_age_days"], step=10)
            bio = st.text_area("Profile Biography (Bio)", value=preset_data["bio"], height=95)
            
        st.markdown("##### 🌐 Network Graph Context")
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
            "Classifier Model to Use for Prediction:",
            list(artifacts["models"].keys()),
            index=0
        )
        
        submit_btn = st.form_submit_button("🔍 Run Account Prediction", use_container_width=True)
        
    if submit_btn:
        # Determine graph context
        G = build_account_graph(list(accounts_df["username"]), edge_list_path="data/illustrative_account_edges.csv")
        graph_ctx = {}
        
        if "Match" in graph_mode and u_name in G:
            # Exact lookup
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
        
        st.markdown("---")
        st.markdown("### 📊 Prediction Results & Explainability")
        
        p_col1, p_col2 = st.columns([1, 1.2])
        
        with p_col1:
            if res["prediction"] == 1:
                st.error(f"🚨 **Prediction: FAKE / BOT ACCOUNT**\n\n**Confidence: {res['confidence']*100:.1f}%**")
            else:
                st.success(f"✅ **Prediction: GENUINE ACCOUNT**\n\n**Confidence: {res['confidence']*100:.1f}%**")
                
            fig_prob = go.Figure(go.Bar(
                x=[res["prob_genuine"], res["prob_fake"]],
                y=["Genuine", "Fake"],
                orientation="h",
                marker=dict(color=["#10b981", "#ef4444"])
            ))
            fig_prob.update_layout(
                title="Model Class Probability Distribution",
                xaxis_title="Probability",
                xaxis=dict(range=[0, 1.0]),
                height=220,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc")
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with p_col2:
            st.markdown("#### 🧠 Influential Decision Signals")
            
            # Metric 1: Follower-to-following ratio
            ratio_status = "⚠️ Highly suspicious (< 0.05)" if res["follower_following_ratio"] < 0.05 else "Normal / Balanced"
            st.markdown(f"• **Follower / Following Ratio**: `{res['follower_following_ratio']:.4f}` — *{ratio_status}*")
            
            # Metric 2: Reputation score
            st.markdown(f"• **Reputation Score**: `{res['reputation_score']:.4f}` (Normal range: 0.5–1.0)")
            
            # Metric 3: Posting rate
            st.markdown(f"• **Activity Intensity**: `{res['posts_per_day']:.2f}` posts per day")
            
            # Metric 4: TF-IDF keywords
            if res["detected_keywords"]:
                keywords_str = ", ".join([f"`{k}`" for k in res["detected_keywords"]])
                st.markdown(f"• **Spam/Bot Text Tokens Matched**: {keywords_str}")
            else:
                st.markdown("• **Spam/Bot Text Tokens Matched**: None (clean/neutral bio)")
                
            # Metric 5: Graph In/Out
            st.markdown(f"• **Network Graph In/Out Ratio**: `{res['graph_in_out_ratio']:.2f}`")


def render_graph_analysis_page(accounts_df, edges_df):
    """View 3: NetworkX and Plotly Graph Analysis"""
    st.markdown("## 🕸️ Network Relationship Analysis")
    st.markdown(
        "Interactive directed network visualization constructed with **NetworkX** and **Plotly**. "
        "Directed edges represent follower connections (`source` $\\to$ `target`)."
    )
    
    st.warning(
        "⚠️ **Synthetic Network Disclaimer**: The connections displayed below represent the illustrative prototype "
        "relationship edge list (`data/illustrative_account_edges.csv`). They are designed to demonstrate graph "
        "centrality and structural analysis, not real scraped social platform data."
    )
    
    if edges_df.empty or accounts_df.empty:
        st.error("Graph edges or accounts data file missing.")
        return
        
    usernames = list(accounts_df["username"])
    G = build_account_graph(usernames, edge_list_path="data/illustrative_account_edges.csv")
    
    # Pre-calculate graph metrics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    pageranks = nx.pagerank(G, alpha=0.85)
    clustering = nx.clustering(G.to_undirected())
    
    # Map label to node
    labels_map = dict(zip(accounts_df["username"], accounts_df["label"]))
    
    # 2D spring layout for Plotly
    pos = nx.spring_layout(G, seed=42, k=0.9)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_colors = []
    
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="rgba(148, 163, 184, 0.45)"),
        hoverinfo="none",
        mode="lines"
    )
    
    # Prepare node traces
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
        color = "#ef4444" if is_fake else "#10b981"
        node_colors.append(color)
        
        # Size based on in-degree
        size = 18 + in_degrees.get(node, 0) * 4
        node_sizes.append(size)
        
        node_info = (
            f"<b>@{node}</b><br>"
            f"Classification: {'FAKE / BOT' if is_fake else 'GENUINE'}<br>"
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
            line=dict(width=2, color="#ffffff")
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Social Network Interaction Topology (Green = Genuine, Red = Fake/Bot)", font=dict(color="#f8fafc")),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            paper_bgcolor="rgba(11, 17, 32, 0.9)",
            plot_bgcolor="rgba(15, 23, 42, 0.9)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=580
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Node Inspector Card
    st.markdown("### 🔬 Inspect Node Metrics")
    selected_node = st.selectbox("Select an account node to inspect detailed graph centrality metrics:", list(G.nodes()))
    
    if selected_node:
        is_fake = labels_map.get(selected_node, 0) == 1
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        
        with col_m1:
            st.metric("Classification", "Fake (1)" if is_fake else "Genuine (0)")
        with col_m2:
            st.metric("In-Degree (Followers)", in_degrees.get(selected_node, 0))
        with col_m3:
            st.metric("Out-Degree (Followees)", out_degrees.get(selected_node, 0))
        with col_m4:
            st.metric("PageRank Score", f"{pageranks.get(selected_node, 0):.4f}")
        with col_m5:
            st.metric("Clustering Coeff", f"{clustering.get(selected_node, 0):.4f}")


def render_model_performance_page(artifacts):
    """View 4: Model Evaluation and Comparison"""
    st.markdown("## 📈 Model Performance & Comparative Evaluation")
    st.markdown("Evaluation metrics computed on the test partition during Phase 3 model training.")
    
    metrics_data = artifacts["metrics"]
    if not metrics_data:
        st.warning("Saved evaluation metrics not found in `outputs/model_evaluation_metrics.json`. Run `python src/model_training.py`.")
        return
        
    results_summary = metrics_data["results_summary"]
    best_model = metrics_data.get("best_model", "Logistic Regression")
    
    st.success(f"🏆 **Selected Best Model**: **{best_model}** (Based on balanced F1-score and generalization simplicity).")
    
    # Tabular Metrics
    models = list(results_summary.keys())
    perf_rows = []
    for m in models:
        r = results_summary[m]
        perf_rows.append({
            "Model": m,
            "Accuracy": f"{r['accuracy']:.2f}",
            "Precision (Fake)": f"{r['precision']:.2f}",
            "Recall (Fake)": f"{r['recall']:.2f}",
            "F1-Score (Fake)": f"{r['f1_score']:.2f}",
            "Train Accuracy": f"{r['train_accuracy']:.2f}"
        })
    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df, use_container_width=True)
    
    # Plotly Grouped Bar Chart
    st.markdown("### 📊 Metrics Comparison Chart")
    plot_data = []
    for m in models:
        r = results_summary[m]
        plot_data.append({"Model": m, "Metric": "Accuracy", "Score": r["accuracy"]})
        plot_data.append({"Model": m, "Metric": "Precision", "Score": r["precision"]})
        plot_data.append({"Model": m, "Metric": "Recall", "Score": r["recall"]})
        plot_data.append({"Model": m, "Metric": "F1-Score", "Score": r["f1_score"]})
        
    bar_df = pd.DataFrame(plot_data)
    fig_bar = px.bar(
        bar_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=["#38bdf8", "#10b981", "#818cf8"],
        title="Comparison Across Classifiers (Test Set)"
    )
    fig_bar.update_layout(
        yaxis=dict(range=[0, 1.15]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc")
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Confusion Matrix Visualization
    st.markdown("### 🔲 Confusion Matrices (Test Partition)")
    cm_cols = st.columns(len(models))
    for idx, m in enumerate(models):
        cm = np.array(results_summary[m]["confusion_matrix"])
        with cm_cols[idx]:
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="True"),
                x=["Genuine (0)", "Fake (1)"],
                y=["Genuine (0)", "Fake (1)"],
                color_continuous_scale="Blues",
                title=f"{m}"
            )
            fig_cm.update_layout(
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc")
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
    st.markdown(
        """
        <div class="info-pill">
            <b>Academic Research Note</b>: The 1.0 (100%) metric scores observed here reflect the small illustrative sample size 
            (N=10, 7 train, 3 test) where patterns are well-separated. In your B.Tech thesis, report these as pipeline verification 
            benchmarks and recommend validating on public corpora like <b>Cresci-2015/2017</b> or <b>TwiBot-20/22</b>.
        </div>
        """,
        unsafe_allow_html=True
    )


def render_dataset_explorer_page(accounts_df):
    """View 5: Dataset Exploration and Interactive Filters"""
    st.markdown("## 📁 Dataset Explorer")
    st.markdown("Inspect, search, and filter the preprocessed accounts dataset.")
    
    if accounts_df.empty:
        st.error("Accounts dataset not found.")
        return
        
    c_f1, c_f2, c_f3 = st.columns([1.5, 1, 1])
    with c_f1:
        search_query = st.text_input("🔍 Search by Username or Bio:", value="")
    with c_f2:
        class_filter = st.selectbox("Filter by Class Label:", ["All Accounts", "Genuine Accounts (0)", "Fake Accounts (1)"])
    with c_f3:
        verified_filter = st.selectbox("Filter by Verification:", ["All", "Verified Only", "Unverified Only"])
        
    # Apply filters
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
        
    st.markdown(f"**Showing {len(filtered_df)} of {len(accounts_df)} accounts:**")
    st.dataframe(filtered_df, use_container_width=True)
    
    st.markdown("### 📊 Distribution Insights")
    c_d1, c_d2 = st.columns(2)
    
    with c_d1:
        # Donut Chart: Class Balance
        label_counts = accounts_df["label"].value_counts().reset_index()
        label_counts["Class"] = label_counts["label"].map({0: "Genuine (0)", 1: "Fake (1)"})
        fig_donut = px.pie(
            label_counts,
            values="count",
            names="Class",
            hole=0.45,
            color="Class",
            color_discrete_map={"Genuine (0)": "#10b981", "Fake (1)": "#ef4444"},
            title="Class Distribution (Genuine vs Fake)"
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with c_d2:
        # Scatter: Followers vs Following
        fig_scatter = px.scatter(
            accounts_df,
            x="following",
            y="followers",
            color=accounts_df["label"].map({0: "Genuine", 1: "Fake"}),
            color_discrete_map={"Genuine": "#10b981", "Fake": "#ef4444"},
            hover_name="username",
            size="posts",
            log_x=True,
            log_y=True,
            title="Followers vs Following (Log Scale, Size = Posts)"
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# =====================================================================
# Main Application Entry Point
# =====================================================================
def main():
    apply_cybersecurity_theme()
    
    # Load resources
    artifacts = load_models_and_artifacts()
    accounts_df, edges_df = load_datasets()
    
    # Sidebar Navigation
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="color: #38bdf8; margin: 0;">🛡️ CyberSentinel</h2>
            <p style="font-size: 0.8rem; color: #94a3b8;">Fake Account Detection System</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    
    nav_selection = st.sidebar.radio(
        "Navigation Menu",
        [
            "🏠 Home Overview",
            "🎯 Account Prediction",
            "🕸️ Network Graph Analysis",
            "📈 Model Performance",
            "📁 Dataset Explorer"
        ],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 System Status")
    if artifacts["is_ready"]:
        st.sidebar.success("✅ Models & Preprocessors Loaded")
        st.sidebar.info(f"Top Model: **{artifacts['metrics'].get('best_model', 'Logistic Regression')}**")
    else:
        st.sidebar.error("❌ Models Missing. Run training.")
        
    st.sidebar.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 25px;">
            <b>B.Tech Project Prototype</b><br>
            Graph + Text + Profile Fusion<br>
            Powered by Streamlit & Scikit-learn
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Page Router
    if nav_selection == "🏠 Home Overview":
        render_home_page(accounts_df, edges_df)
    elif nav_selection == "🎯 Account Prediction":
        render_prediction_page(artifacts, accounts_df, edges_df)
    elif nav_selection == "🕸️ Network Graph Analysis":
        render_graph_analysis_page(accounts_df, edges_df)
    elif nav_selection == "📈 Model Performance":
        render_model_performance_page(artifacts)
    elif nav_selection == "📁 Dataset Explorer":
        render_dataset_explorer_page(accounts_df)


if __name__ == "__main__":
    main()
