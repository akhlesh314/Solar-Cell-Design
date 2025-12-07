import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO
from openai import OpenAI

# ============================================================
# Load API Key from Render Environment Variables
# ============================================================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "❌ OPENAI_API_KEY not found.\n\n"
        "Set it in Render → Environment → Environment Variables."
    )
    st.stop()

client = OpenAI(api_key=api_key)

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(
    page_title="Agentic AI Data Analysis Bot",
    layout="wide"
)

st.title("📊 Agentic AI Data Analysis Bot")

st.write(
    """
    Upload a CSV dataset, and this agent will build an **end-to-end machine learning plan**  
    including **data cleaning**, **feature engineering**, **model selection**,  
    **Python code**, and **evaluation strategy**—all generated using a  
    PEACE-structured LLM prompt.
    """
)

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

# ============================================================
# Process CSV Upload
# ============================================================
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # ---- Preview ----
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ---- Show df.info() ----
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    # ---- Summary ----
    st.subheader("📊 Dataset Summary")
    st.write(df.describe(include="all"))

    # ============================================================
    # Build the PEACE Prompt
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
You are a senior data scientist. Analyze the uploaded dataset and generate
a complete machine-learning pipeline proposal.

E — EXPECTATIONS:
Provide:
1. Data cleaning steps
2. Feature engineering strategy
3. Determination of task type (regression/classification).  
   If unclear, state your assumption clearly.
4. Recommended ML algorithm(s) and rationale
5. Full Python code using:
   - pandas
   - scikit-learn
   - matplotlib
6. Explanation of evaluation metrics and interpretation

A — ACTIONS:
- Examine dataset summary
- Identify transformations needed
- Recommend the ML approach
- Produce complete executable Python code
- Provide metrics explanation

C — CONSTRAINTS:
- DO NOT hallucinate columns, features, or values
- Only use columns that appear in the dataset
- If unsure about task type, state the assumption
- Do not fabricate metric values; only provide code

E — EVALUATION:
Your output must be:
- Technically correct
- Logically consistent
- Structured into sections:
    * Data Cleaning
    * Feature Engineering
    * Model Selection
    * Full Python Code
    * Evaluation Strategy
- Reproducible by a graduate-level student

===========================
DATASET SUMMARY BELOW
===========================
{dataset_summary}
"""

    # ============================================================
    # OpenAI API Call
    # ============================================================
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": peace_prompt
            }],
            temperature=0.2,
        )

        # FIXED: Use attribute access, NOT dictionary access
        report = response.choices[0].message.content

    except Exception as e:
        st.error(f"❌ OpenAI API Error:\n{e}")
        st.stop()

    # ============================================================
    # Display Results
    # ============================================================
    st.subheader("📘 Full AI-Generated Analysis Report")
    st.write(report)

    # ============================================================
    # Download Report
    # ============================================================
    st.download_button(
        label="📥 Download Report as TXT",
        data=report,
        file_name="analysis_report.txt",
        mime="text/plain"
    )

else:
    st.info("👆 Upload a CSV file to begin.")
