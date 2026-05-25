"""
Multimodal Earnings Call Intelligence Dashboard — Phase 5 Final

Interactive analyst terminal with LIVE ANALYSIS:
- Paste transcript text, upload file, or provide URL
- Real-time FinBERT sentiment + uncertainty + specificity analysis
- Model prediction (BUY/SELL/HOLD)
"""

import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multimodal Analyst Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2227; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    processed = PROJECT_ROOT / "data" / "processed"
    outputs_eval = PROJECT_ROOT / "outputs" / "evaluation"
    outputs_root = PROJECT_ROOT / "outputs"

    df = None
    for fname in ["text_market_dataset.parquet", "multimodal_dataset.parquet"]:
        p = processed / fname
        if p.exists():
            try:
                df = pl.read_parquet(p).to_pandas()
            except Exception as e:
                st.error(f"Failed to load {fname}: {e}")
            break

    predictions = None
    pred_path = outputs_eval / "model_predictions.parquet"
    if pred_path.exists():
        try:
            predictions = pl.read_parquet(pred_path).to_pandas()
        except Exception as e:
            st.error(f"Failed to load predictions: {e}")

    backtest = None
    bt_path = outputs_eval / "backtest_results.json"
    if bt_path.exists():
        with open(bt_path) as f:
            backtest = json.load(f)

    baselines = None
    bc_path = outputs_root / "baseline_comparison.json"
    if bc_path.exists():
        with open(bc_path) as f:
            baselines = json.load(f)

    return df, backtest, predictions, baselines


# ---------------------------------------------------------------------------
# Live Analysis Engine
# ---------------------------------------------------------------------------

@st.cache_resource
def load_sentiment_model():
    """Load FinBERT for live inference."""
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None,
            device="cpu",
        )
        return pipe
    except Exception as e:
        st.error(f"Could not load FinBERT: {e}")
        return None


@st.cache_resource
def load_spacy_model():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def analyze_text_live(text: str, sentiment_pipe, nlp_spacy) -> dict:
    """Run full text analysis on raw transcript text."""
    if not text or len(text.strip()) < 10:
        return None

    # Split into segments (by paragraph or speaker turns)
    raw_segments = re.split(r'\n{2,}|\r\n{2,}', text.strip())
    raw_segments = [s.strip() for s in raw_segments if len(s.strip()) > 20]
    if not raw_segments:
        raw_segments = [text.strip()]

    # --- Sentiment ---
    sentiments = []
    if sentiment_pipe:
        for seg in raw_segments:
            truncated = seg[:512]
            try:
                out = sentiment_pipe(truncated, truncation=True, max_length=512)
                scores = {d["label"]: d["score"] for d in out[0]}
                sentiments.append(scores.get("positive", 0) - scores.get("negative", 0))
            except Exception:
                sentiments.append(0.0)
    else:
        sentiments = [0.0] * len(raw_segments)

    # --- Uncertainty & Hedging ---
    uncertainty_terms = {
        "may", "might", "could", "possibly", "uncertain", "unclear", "approximately",
        "roughly", "perhaps", "potential", "likely", "unlikely", "estimate", "expect",
        "anticipate", "believe", "assume", "appear", "seem", "suggest", "indicate",
        "risk", "volatile", "fluctuate", "unpredictable", "contingent",
    }
    hedging_phrases = [
        "to some extent", "we believe", "it is possible", "in our view",
        "we expect", "going forward", "at this point", "as we see it",
        "we think", "it appears", "more or less", "we anticipate",
    ]

    unc_scores = []
    hedge_scores = []
    for seg in raw_segments:
        tokens = seg.lower().split()
        n = max(len(tokens), 1)
        unc_count = sum(1 for t in tokens if t.strip(".,;:!?") in uncertainty_terms)
        unc_scores.append(unc_count / n)
        seg_lower = seg.lower()
        h_count = sum(1 for p in hedging_phrases if p in seg_lower)
        hedge_scores.append(h_count / n)

    # --- Specificity & Complexity ---
    spec_scores = []
    complexity_scores = []
    fls_phrases = ["we expect", "going forward", "next quarter", "looking ahead",
                   "we anticipate", "our outlook", "future growth", "we plan"]

    for seg in raw_segments:
        tokens = seg.split()
        n = max(len(tokens), 1)
        nums = sum(1 for t in tokens if re.search(r'\d', t))
        entities = 0
        if nlp_spacy:
            try:
                doc = nlp_spacy(seg[:5000])
                entities = len(doc.ents)
            except Exception:
                pass
        spec_scores.append(min(1.0, (nums + entities) / n))
        fls_count = sum(1 for p in fls_phrases if p in seg.lower())
        complexity_scores.append(fls_count / n)

    # --- Aggregate ---
    avg = lambda lst: sum(lst) / max(len(lst), 1)
    result = {
        "n_segments": len(raw_segments),
        "total_chars": sum(len(s) for s in raw_segments),
        "sentiment_mean": avg(sentiments),
        "sentiment_min": min(sentiments) if sentiments else 0,
        "sentiment_max": max(sentiments) if sentiments else 0,
        "uncertainty_mean": avg(unc_scores),
        "hedging_mean": avg(hedge_scores),
        "specificity_mean": avg(spec_scores),
        "forward_looking_mean": avg(complexity_scores),
        "sentiments_per_segment": sentiments,
        "segments": raw_segments,
    }

    # --- Signal Generation ---
    pressure = result["uncertainty_mean"] - result["sentiment_mean"]
    if result["sentiment_mean"] > 0.15 and result["uncertainty_mean"] < 0.05:
        result["signal"] = "BUY"
        result["confidence"] = min(95, abs(result["sentiment_mean"]) * 150)
    elif result["sentiment_mean"] < -0.1 or result["uncertainty_mean"] > 0.08:
        result["signal"] = "SELL"
        result["confidence"] = min(95, (abs(result["sentiment_mean"]) + result["uncertainty_mean"]) * 120)
    else:
        result["signal"] = "HOLD"
        result["confidence"] = 30 + abs(result["sentiment_mean"]) * 50

    result["pressure_score"] = float(max(0, min(1, (pressure + 0.5))))
    return result


def fetch_text_from_url(url: str) -> str:
    """Download transcript text from a URL."""
    try:
        import requests
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', resp.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

df, backtest, predictions, baselines = load_data()

st.sidebar.title("📈 Analyst Terminal")
st.sidebar.markdown("---")
view = st.sidebar.radio("View Mode", [
    "🔬 Live Analysis",
    "🛰️ Signal Monitor",
    "📊 Performance",
    "🔍 Company Deep Dive",
    "🏆 Model Performance",
])

st.sidebar.markdown("---")
if df is not None:
    n_calls = len(df)
    n_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
    st.sidebar.info(f"📊 {n_calls} Calls | {n_tickers} Tickers")
else:
    st.sidebar.warning("No dataset loaded yet")

# ---------------------------------------------------------------------------
# View: Live Analysis (NEW)
# ---------------------------------------------------------------------------

if view == "🔬 Live Analysis":
    st.title("🔬 Live Earnings Call Analyzer")
    st.markdown("Paste transcript text, upload a file, or provide a URL — get instant AI analysis.")

    input_method = st.radio("Input Method", ["📝 Paste Text", "📁 Upload File", "🔗 URL"], horizontal=True)

    transcript_text = ""

    if input_method == "📝 Paste Text":
        transcript_text = st.text_area(
            "Paste earnings call transcript here",
            height=300,
            placeholder="Good afternoon. Welcome to the Q3 2024 Earnings Call. I'd like to turn the call over to the CEO..."
        )

    elif input_method == "📁 Upload File":
        uploaded = st.file_uploader("Upload transcript (.txt, .md, .csv)", type=["txt", "md", "csv", "text"])
        if uploaded:
            transcript_text = uploaded.read().decode("utf-8", errors="replace")
            st.success(f"Loaded {len(transcript_text):,} characters from {uploaded.name}")

    elif input_method == "🔗 URL":
        url = st.text_input("Enter URL to earnings call transcript")
        if url and st.button("🌐 Fetch Transcript"):
            with st.spinner("Downloading..."):
                transcript_text = fetch_text_from_url(url)
                if transcript_text.startswith("ERROR:"):
                    st.error(transcript_text)
                    transcript_text = ""
                else:
                    st.success(f"Fetched {len(transcript_text):,} characters")

    # --- Run Analysis ---
    if transcript_text and len(transcript_text.strip()) > 50:
        if st.button("🚀 Analyze Now", type="primary") or st.session_state.get("_auto_analyze"):
            with st.spinner("Running FinBERT sentiment + NLP analysis..."):
                sent_pipe = load_sentiment_model()
                nlp_spacy = load_spacy_model()
                result = analyze_text_live(transcript_text, sent_pipe, nlp_spacy)

            if result is None:
                st.error("Text too short for analysis.")
            else:
                st.markdown("---")

                # --- Signal Banner ---
                signal = result["signal"]
                color = {"BUY": "#2ea043", "SELL": "#f85149", "HOLD": "#d29922"}[signal]
                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[signal]
                st.markdown(
                    f'<div style="background:{color}22; border:2px solid {color}; border-radius:12px; '
                    f'padding:20px; text-align:center; margin-bottom:20px;">'
                    f'<h1 style="color:{color}; margin:0;">{emoji} {signal}</h1>'
                    f'<p style="color:#e6edf3; font-size:18px; margin:5px 0;">Confidence: {result["confidence"]:.0f}%</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # --- Key Metrics ---
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Sentiment", f"{result['sentiment_mean']:.3f}")
                col2.metric("Uncertainty", f"{result['uncertainty_mean']:.4f}")
                col3.metric("Hedging", f"{result['hedging_mean']:.4f}")
                col4.metric("Specificity", f"{result['specificity_mean']:.3f}")
                col5.metric("Pressure", f"{result['pressure_score']:.2f}")

                st.markdown("### 📊 Segment-Level Sentiment Breakdown")

                # Sentiment per segment chart
                seg_labels = [f"Seg {i+1}" for i in range(len(result["sentiments_per_segment"]))]
                seg_colors = ["#2ea043" if s > 0 else "#f85149" for s in result["sentiments_per_segment"]]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=seg_labels,
                    y=result["sentiments_per_segment"],
                    marker_color=seg_colors,
                    text=[f"{s:.2f}" for s in result["sentiments_per_segment"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#1e2227",
                    yaxis_title="Sentiment Score",
                    xaxis_title="Segment",
                    title="Sentiment per Transcript Segment",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Pressure Gauge ---
                col_left, col_right = st.columns([1, 2])
                with col_left:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["pressure_score"],
                        title={"text": "Executive Pressure"},
                        gauge={
                            "axis": {"range": [0, 1]},
                            "bar": {"color": "#f85149" if result["pressure_score"] > 0.6 else "#2ea043"},
                            "steps": [
                                {"range": [0, 0.4], "color": "#1e2227"},
                                {"range": [0.4, 0.7], "color": "#30363d"},
                                {"range": [0.7, 1.0], "color": "#6e7681"},
                            ],
                        },
                    ))
                    fig_gauge.update_layout(template="plotly_dark", height=280,
                                            margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col_right:
                    # Feature radar
                    categories = ["Sentiment", "Certainty", "Specificity", "Forward-Looking", "Low Hedging"]
                    values = [
                        max(0, (result["sentiment_mean"] + 1) / 2),
                        max(0, 1 - result["uncertainty_mean"] * 10),
                        result["specificity_mean"],
                        min(1, result["forward_looking_mean"] * 20),
                        max(0, 1 - result["hedging_mean"] * 20),
                    ]
                    fig_radar = go.Figure(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        fill="toself",
                        fillcolor="rgba(88, 166, 255, 0.2)",
                        line=dict(color="#58a6ff", width=2),
                    ))
                    fig_radar.update_layout(
                        template="plotly_dark",
                        polar=dict(bgcolor="#1e2227",
                                   radialaxis=dict(range=[0, 1], showticklabels=False)),
                        title="Confidence Profile",
                        height=300,
                        margin=dict(l=40, r=40, t=50, b=30),
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                # --- Raw data expander ---
                with st.expander("📄 View Processed Segments"):
                    for i, seg in enumerate(result["segments"][:20]):
                        sent = result["sentiments_per_segment"][i] if i < len(result["sentiments_per_segment"]) else 0
                        icon = "🟢" if sent > 0.1 else ("🔴" if sent < -0.1 else "⚪")
                        st.markdown(f"**{icon} Segment {i+1}** (sentiment: {sent:.3f})")
                        st.caption(seg[:300] + ("..." if len(seg) > 300 else ""))

                # Export
                st.download_button(
                    "📥 Download Analysis JSON",
                    data=json.dumps({k: v for k, v in result.items() if k != "segments"}, indent=2, default=str),
                    file_name="analysis_result.json",
                    mime="application/json",
                )

    elif transcript_text:
        st.warning("Please provide at least 50 characters of text.")

# ---------------------------------------------------------------------------
# View: Signal Monitor
# ---------------------------------------------------------------------------

elif view == "🛰️ Signal Monitor":
    st.title("🛰️ Signal Monitor")

    if predictions is not None and len(predictions) > 0:
        display_df = predictions.copy()
        display_df["Signal"] = display_df["signal"].map({1: "BUY", -1: "SELL", 0: "HOLD"})
        display_df["Confidence"] = (abs(display_df["pred_proba"] - 0.5) * 200).clip(0, 95).round(1)
        display_df["Correct"] = display_df["pred_class"] == display_df["actual_direction"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(display_df))
        c2.metric("Buy", (display_df["Signal"] == "BUY").sum())
        c3.metric("Sell", (display_df["Signal"] == "SELL").sum())
        c4.metric("Accuracy", f"{display_df['Correct'].mean():.1%}")

        st.dataframe(
            display_df[["call_id", "ticker", "date", "Signal", "Confidence", "Correct"]].sort_values("date", ascending=False),
            use_container_width=True, hide_index=True,
        )
    elif df is not None:
        st.info("No model predictions yet. Run the backtest pipeline first.")
        st.dataframe(df[["call_id", "ticker", "call_date"]].head(50) if "ticker" in df.columns else df.head(50),
                     use_container_width=True, hide_index=True)
    else:
        st.warning("No data available.")

# ---------------------------------------------------------------------------
# View: Performance
# ---------------------------------------------------------------------------

elif view == "📊 Performance":
    st.title("📊 Strategy Performance")

    if backtest and backtest.get("methodology", "").startswith("walk_forward_backtest"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Strategy Return", f"{backtest['total_strategy_return']*100:.2f}%",
                   delta=f"{backtest['alpha']*100:+.2f}% alpha")
        c2.metric("Sharpe Ratio", f"{backtest['annualized_sharpe']:.2f}")
        c3.metric("Hit Rate", f"{backtest['hit_rate']*100:.1f}%")
        c4.metric("Max Drawdown", f"{backtest['max_drawdown']*100:.1f}%")

        sig = backtest.get("statistically_significant", False)
        pval = backtest.get("p_value", 1.0)
        if sig:
            st.success(f"✅ Statistically significant (p={pval:.4f})")
        else:
            st.warning(f"⚠️ NOT statistically significant (p={pval:.4f})")

        trades = pd.DataFrame(backtest["trades"])
        active = trades[trades["signal"] != 0].copy()
        if len(active) > 0:
            active["cum_strategy"] = (1 + active["pnl"]).cumprod() - 1
            active["cum_benchmark"] = (1 + active["actual_ret"]).cumprod() - 1
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(active))), y=active["cum_strategy"]*100,
                                     name="Strategy", line=dict(color="#2ea043", width=3)))
            fig.add_trace(go.Scatter(x=list(range(len(active))), y=active["cum_benchmark"]*100,
                                     name="Benchmark", line=dict(color="#30363d", width=2, dash="dash")))
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                              yaxis_title="Return (%)", xaxis_title="Trade #")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run `python -m src.evaluation.backtesting` first.")

# ---------------------------------------------------------------------------
# View: Company Deep Dive
# ---------------------------------------------------------------------------

elif view == "🔍 Company Deep Dive":
    st.title("🔍 Company Deep Dive")
    if df is not None and "ticker" in df.columns:
        ticker = st.selectbox("Select Company", sorted(df["ticker"].unique()))
        row = df[df["ticker"] == ticker].iloc[0]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"### {ticker}")
            if "call_date" in row: st.write(f"**Date:** {row['call_date']}")

            sent_cols = [c for c in df.columns if "sentiment" in c.lower() and "mean" in c.lower()]
            if sent_cols:
                val = float(row[sent_cols[0]])
                fig = go.Figure(go.Indicator(mode="gauge+number", value=val,
                    title={"text": "Sentiment"}, gauge={"axis": {"range": [-1, 1]},
                    "bar": {"color": "#f85149" if val < -0.2 else "#2ea043"}}))
                fig.update_layout(template="plotly_dark", height=250, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            feats = {}
            for prefix, label in [("sentiment","Sentiment"),("uncertainty","Uncertainty"),
                                   ("hedging","Hedging"),("specificity","Specificity")]:
                cols = [c for c in df.columns if prefix in c.lower() and "mean" in c.lower()]
                if cols:
                    v = float(row[cols[0]])
                    if pd.notna(v): feats[label] = v
            if feats:
                fig = px.bar(x=list(feats.keys()), y=list(feats.values()),
                             color=list(feats.values()), color_continuous_scale="RdYlGn",
                             title="Feature Profile")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available.")

# ---------------------------------------------------------------------------
# View: Model Performance
# ---------------------------------------------------------------------------

elif view == "🏆 Model Performance":
    st.title("🏆 Model Performance Report")

    if baselines:
        cls_results = [r for r in baselines if r["task"] == "direction_classification"]
        if cls_results:
            st.markdown("#### Direction Prediction Accuracy")
            rows = []
            for r in cls_results:
                acc = r["accuracy"]
                if isinstance(acc, dict):
                    rows.append({"Model": r["model"], "Accuracy": f"{acc['mean']:.1%}",
                                 "95% CI": f"[{acc['ci_low']:.1%}, {acc['ci_high']:.1%}]",
                                 "Samples": r.get("total_samples", "?")})
                else:
                    rows.append({"Model": r["model"], "Accuracy": f"{acc:.1%}", "95% CI": "N/A"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Show plots
    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    if plots_dir.exists():
        st.markdown("### Visualizations")
        plot_files = sorted(plots_dir.glob("*.png"))
        if plot_files:
            cols = st.columns(2)
            for i, pf in enumerate(plot_files):
                with cols[i % 2]:
                    st.image(str(pf), caption=pf.stem.replace("_", " ").title(), use_container_width=True)

st.markdown("---")
st.caption("Powered by Multimodal Earnings Pipeline | Walk-Forward Validated")
