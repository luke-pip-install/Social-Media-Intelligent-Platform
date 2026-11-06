import streamlit as st

from common import configure, load_data

configure("Signal Desk")
df, source = load_data()

st.title("Signal Desk")
st.caption("A local social listening workspace inspired by the analysis in NotebookName3.ipynb.")

if source == "demo":
    st.info("Showing illustrative demo posts. Upload a CSV or JSON export in the sidebar to analyze your own data.")
else:
    st.success(f"Loaded {len(df):,} posts from {source}. Your file stays in this local Streamlit session.")

st.markdown("### Choose an analysis page")
left, right = st.columns(2)
with left:
    st.subheader("Basic Techniques")
    st.write("Sentiment, hashtags, trends, entities, and topic clusters.")
    st.page_link("pages/1_Basic_Techniques.py", label="Open Basic Techniques →")
with right:
    st.subheader("Advanced Techniques")
    st.write("News context, a trend-to-brand brief, and multilingual crisis triage.")
    st.page_link("pages/2_Advanced_Techniques.py", label="Open Advanced Techniques →")

st.divider()
st.caption("This app analyzes uploaded or demo data. Connecting a platform API is a separate integration step. Classifier output and keyword flags need human review.")
