from __future__ import annotations

import hashlib
import re
import streamlit as st

from monitor import demo_frame, read_upload


def configure(title: str):
    st.set_page_config(page_title=f"{title} · Signal Desk", page_icon="📡", layout="wide")


def load_data():
    st.sidebar.title("📡 Signal Desk")
    st.sidebar.caption("LOCAL SOCIAL LISTENING")
    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Import a post export", type=["csv", "json", "jsonl"], help="CSV with a text/tweet/content column, or headerless Sentiment140 CSV. JSON and JSONL are also supported.")
    if uploaded is not None:
        payload = uploaded.getvalue()
        marker = (uploaded.name, hashlib.md5(payload).hexdigest())
        if st.session_state.get("upload_marker") != marker:
            try:
                st.session_state["monitor_df"] = read_upload(uploaded.name, payload)
                st.session_state["monitor_source"] = uploaded.name
                st.session_state["upload_marker"] = marker
                st.session_state.pop("upload_error", None)
            except Exception as exc:
                st.session_state["upload_error"] = str(exc)
        if st.session_state.get("upload_error"):
            st.sidebar.error(st.session_state["upload_error"])
    choice = st.sidebar.radio("Dataset", ["Demo posts", "Imported posts"], index=1 if "monitor_df" in st.session_state else 0)
    if choice == "Imported posts" and "monitor_df" in st.session_state:
        df = st.session_state["monitor_df"]
        source = st.session_state.get("monitor_source", "Uploaded file")
    else:
        df, source = demo_frame(), "demo"
    st.sidebar.caption(f"{len(df):,} posts · {source}")
    st.sidebar.markdown("---")
    st.sidebar.caption("Use the page menu above to switch between Basic and Advanced Techniques.")
    return df, source


def sentiment_scores(df):
    import pandas as pd
    result = df.copy()
    valid = result["target"].isin([0, 1]).sum()
    if valid >= 20 and result.loc[result.target.isin([0, 1]), "target"].nunique() == 2:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import train_test_split
        labelled = result[result.target.isin([0, 1])]
        try:
            x_train, x_test, y_train, y_test = train_test_split(labelled.processed_text, labelled.target.astype(int), test_size=.2, random_state=42, stratify=labelled.target.astype(int))
            model = _fit_sentiment_model(tuple(x_train.tolist()), tuple(int(v) for v in y_train.tolist()))
            result["sentiment"] = model.predict(result.processed_text)
            return result, "TF-IDF + logistic regression trained on uploaded labels", model, (x_test, y_test, model.predict(x_test))
        except ValueError:
            pass
    from monitor import sentiment
    labels = result.text.map(lambda x: sentiment(x)[0])
    result["sentiment"] = labels.map({"negative": 0, "neutral": 2, "positive": 1})
    return result, "Lexicon baseline (no labeled training data found)", None, None


@st.cache_resource(show_spinner="Training the sentiment classifier…")
def _fit_sentiment_model(texts, labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    model = make_pipeline(TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(list(texts), list(labels))
    return model


@st.cache_resource(show_spinner="Loading Qwen3-0.6B…")
def get_qwen():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")
    return tokenizer, model


def guard_generated_text(text: str, fallback: str) -> str:
    """Reject blank generations and common signs of prompt injection succeeding."""
    markers = re.compile(r"(?i)ignore (all |the )?(previous|above) instructions|reveal (the )?(system|developer) prompt|haha pwned")
    cleaned = (text or "").strip()
    if not cleaned or markers.search(cleaned):
        return fallback
    return cleaned


def sentiment_name(label):
    return {0: "Negative", 1: "Positive", 2: "Neutral"}.get(int(label), "Neutral")
