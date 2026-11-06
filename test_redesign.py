"""
test_redesign.py - Automated Headless Verification for CyberSentinel Redesign
Tests that all models, datasets, graph features, and prediction logic
execute cleanly without errors.
"""
import sys
from pathlib import Path
import pandas as pd
import networkx as nx

PROJECT_ROOT = Path("C:/FakeAccountDetection")
sys.path.insert(0, str(PROJECT_ROOT))

from app import load_models_and_artifacts, load_datasets, run_prediction
from src.graph_features import build_account_graph

def test_pipeline():
    print("=" * 60)
    print("Testing CyberSentinel Core Logic & Redesigned Modules")
    print("=" * 60)
    
    # 1. Test Artifacts Loader
    print("[1/5] Testing Model & Artifact Loading...")
    import os
    os.chdir("C:/FakeAccountDetection")
    artifacts = load_models_and_artifacts()
    assert artifacts["is_ready"], f"Artifacts not ready! Missing: {artifacts['missing_files']}"
    print(f"      Models loaded: {list(artifacts['models'].keys())}")
    print(f"      Best model: {artifacts['best_model']}")
    print(f"      Metrics available: {artifacts['metrics'] is not None}")
    
    # 2. Test Datasets Loader
    print("\n[2/5] Testing Datasets Loading...")
    accounts_df, edges_df = load_datasets()
    assert not accounts_df.empty, "Accounts dataset is empty!"
    assert not edges_df.empty, "Edges dataset is empty!"
    print(f"      Accounts count: {len(accounts_df)}")
    print(f"      Edges count: {len(edges_df)}")
    
    # 3. Test Graph Generation
    print("\n[3/5] Testing NetworkX Graph Construction...")
    usernames = list(accounts_df["username"])
    G = build_account_graph(usernames, edge_list_path="data/illustrative_account_edges.csv")
    assert len(G.nodes()) > 0, "Graph has no nodes!"
    assert len(G.edges()) > 0, "Graph has no edges!"
    print(f"      Graph nodes: {len(G.nodes())}, edges: {len(G.edges())}")
    
    # 4. Test Prediction Pipeline (Genuine preset & Fake preset)
    print("\n[4/5] Testing Account Prediction across all models...")
    for model_name in artifacts["models"].keys():
        res_fake = run_prediction(
            username="bot_spammer_99",
            followers=10,
            following=4000,
            posts=1,
            account_age_days=3,
            bio="Click my bio link to earn free cash fast",
            is_verified=False,
            graph_context={"in_degree": 0.0, "out_degree": 5.0, "pagerank": 0.04, "betweenness": 0.0, "clustering": 0.0, "weak_comp": 10.0, "strong_comp": 1.0},
            model_name=model_name,
            artifacts=artifacts
        )
        print(f"      [{model_name}] Fake test prediction: {res_fake['prediction']} ({res_fake['label_name']}) | Confidence: {res_fake['confidence']*100:.1f}%")
        assert res_fake["prediction"] in [0, 1]
        
        res_genuine = run_prediction(
            username="tech_researcher",
            followers=5500,
            following=420,
            posts=850,
            account_age_days=1200,
            bio="Senior AI researcher and engineer working on cybersecurity.",
            is_verified=True,
            graph_context={"in_degree": 6.0, "out_degree": 2.0, "pagerank": 0.15, "betweenness": 0.05, "clustering": 0.4, "weak_comp": 10.0, "strong_comp": 2.0},
            model_name=model_name,
            artifacts=artifacts
        )
        print(f"      [{model_name}] Genuine test prediction: {res_genuine['prediction']} ({res_genuine['label_name']}) | Confidence: {res_genuine['confidence']*100:.1f}%")
        assert res_genuine["prediction"] in [0, 1]

    # 5. Test Filtering Logic (Dataset Explorer)
    print("\n[5/5] Testing Dataset Filtering Logic...")
    fake_subset = accounts_df[accounts_df["label"] == 1]
    genuine_subset = accounts_df[accounts_df["label"] == 0]
    print(f"      Genuine accounts: {len(genuine_subset)}, Fake accounts: {len(fake_subset)}")
    assert len(fake_subset) > 0 and len(genuine_subset) > 0
    
    print("\n" + "=" * 60)
    print("ALL CORE FUNCTIONALITY & PREDICTION PIPELINES VERIFIED!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(test_pipeline())
