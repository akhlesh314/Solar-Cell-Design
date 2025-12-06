import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO
from openai import OpenAI

# ============================================================
# 1. Load API Key from Render Environment Variables
# ============================================================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "❌ OPENAI_API_KEY not found.\n\n"
        "You must set it in Render → Environment → Environment Variables."
    )
    st.stop()

client = OpenAI(api_key=api_key)

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Agentic AI Data Analysis Bot", layout="wide")
st.title("📊 Agentic AI Data Analysis Bot")

st.write(
    """
    Upload a CSV file and let the agent produce an end-to-end ML analysis pipeline  
    including cleaning, feature engineering, model selection, Python code, and evaluation.
    """
)

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    # ============================================================
    # Read CSV
    # ============================================================
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Display df.info()
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    # Summary statistics
    st.subheader("📊 Dataset Summary")
    st.write(df.describe(include="all"))

    # ============================================================
    # Construct PEACE Prompt
    # ============================================================
    st.subheader("🤖 Running Agentic Dataset Analysis…")

    dataset_summary = f"""
    DATA HEAD:
    {df.head().to_string()}

    DATA INFO:
    {info_str}

    SUMMARY:
    {df.describe(include="all").to_string()}
    """

    peace_prompt = f"""
P — PURPOSE:
You are a senior data scientist. Analyze the uploaded dataset thoroughly and produce
a full ML workflow ready for implementation.

E — EXPECTATIONS:
Provide:
1. Data cleaning steps  
2. Feature engineering strategy  
3. Determine if the task is regression or classification  
4. Select appropriate ML models  
5. Provide COMPLETE Python code using:
   - pandas
   - scikit-learn
   - matplotlib
6. Explain evaluation metrics and how to interpret them  
7. Ensure the explanation is academically sound and reproducible.

A — ACTIONS:
- Examine the dataset summary provided
- Reason about transformations
- Identify the target variable (state assumptions if unsure)
- Suggest the best algorithms
- Generate full executable Python code
- Provide evaluation guidance

C — CONSTRAINTS:
- DO NOT hallucinate columns or features
- Only reference real columns in the dataset
- Do not assume the task type; state an assumption if needed
- Do not fabricate metric values — only provide code

E — EVALUATION:
Your final output must be:
- Correct and logically consistent
- Structured into sections:
    * Cleaning
    * Feature Engineering
    * Model Selection
    * Full Python Code
    * Evaluation Strategy
- Reproducible by a graduate student in Python

===========================
DATASET SUMMARY BELOW
===========================
{dataset_summary}
"""

    # ============================================================
    # Call OpenAI
    # ============================================================
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": peace_prompt}],
            temperature=0.2,
        )
        report = response.choices[0].message["content"]

    except Exception as e:
        st.error(f"❌ OpenAI API Error: {e}")
        st.stop()

    st.subheader("📘 Full AI-Generated Analysis Report")
    st.write(report)

    # ============================================================
    # Option to Download Report
    # ============================================================
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="analysis_report.txt",
        mime="text/plain"
    )

else:
    st.info("👆 Upload a CSV file to begin.")
