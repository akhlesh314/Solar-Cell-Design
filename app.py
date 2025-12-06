"""
Solar-Cell-Design Bot (Streamlit + OpenAI)

Features:
- CSV uploader
- Dataset preview (df.head(), df.info())
- Rich dataset summary via pandas
- Agentic workflow using OpenAI with PEACE prompt framework:
    P — Purpose
    E — Expectations
    A — Actions
    C — Constraints
    E — Evaluation
- Produces: cleaning strategy, feature engineering suggestions, model selection,
  full Python ML pipeline code (pandas + scikit-learn + matplotlib), evaluation explanation
- Full report shown in-streamlit and downloadable as .txt

Usage:
- Deploy to Render (set OPENAI_API_KEY in Environment variables or add it to Streamlit secrets)
- Streamlit will load the key from st.secrets["OPENAI_API_KEY"]
"""

import io
import textwrap
import traceback
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Use the new OpenAI Python SDK pattern requested by the user
# Note: package name in requirements.txt is 'openai'
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # We'll handle missing import at runtime

# -------------------------
# Streamlit app UI
# -------------------------
st.set_page_config(page_title="Agentic AI Data Analysis Bot", layout="wide")

st.title("Agentic AI Data Analysis Bot")
st.markdown(
    """
Upload a CSV dataset and the agent will:
- Summarize the dataset
- Propose a cleaning & feature-engineering pipeline
- Recommend models (classification/regression)
- Produce runnable Python training code (pandas + scikit-learn + matplotlib)
- Explain evaluation metrics

All LLM interactions use the **P.E.A.C.E. Prompt Framework** (Purpose, Expectations, Actions, Constraints, Evaluation).
"""
)

# Sidebar: API key check and options
st.sidebar.header("Settings & API")
st.sidebar.write(
    "The app reads your OpenAI API key from `st.secrets['OPENAI_API_KEY']`. "
    "On Render, set an environment variable `OPENAI_API_KEY` and add it to Streamlit secrets as described in the README."
)

show_raw_llm = st.sidebar.checkbox("Show raw LLM output", value=False)
model_choice = st.sidebar.text_input(
    "LLM model name (optional)", value="gpt-4o-mini", help="If you know a specific model name, set it here."
)

# Upload CSV
st.header("1) Upload CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Helper functions
def safe_read_csv(file) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(file)
    except Exception:
        try:
            # fallback: try with latin-1
            file.seek(0)
            return pd.read_csv(file, encoding="latin-1")
        except Exception:
            return None

def df_info_to_string(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

def make_dataset_summary(df: pd.DataFrame) -> str:
    # Basic summary: shape, dtypes, missing, head, describe, unique counts, correlations for numerics
    pieces = []
    pieces.append(f"Shape: {df.shape}")
    pieces.append("\nColumn dtypes:\n" + df.dtypes.astype(str).to_string())
    miss = df.isnull().sum()
    pieces.append("\nMissing values per column:\n" + miss.to_string())
    pieces.append("\nSample (first 5 rows):\n" + df.head(5).to_csv(index=False))
    try:
        desc = df.describe(include="all").transpose()
        pieces.append("\nSummary statistics (describe):\n" + desc.to_string())
    except Exception:
        pieces.append("\nSummary statistics (describe) failed or not applicable.")
    # unique counts (limited)
    unique_counts = df.nunique(dropna=False).sort_values()
    pieces.append("\nUnique value counts (per column):\n" + unique_counts.to_string())
    # correlations for numeric subset
    num = df.select_dtypes(include=["number"])
    if not num.empty:
        try:
            corr = num.corr()
            pieces.append("\nNumeric correlation matrix (top correlations):")
            pieces.append(corr.to_string())
        except Exception:
            pieces.append("\nCorrelation computation failed.")
    else:
        pieces.append("\nNo numeric columns for correlation.")
    return "\n\n".join(pieces)

def build_peace_prompt(dataset_summary_text: str, column_list: str, sample_head_csv: str) -> str:
    """
    Build the PEACE prompt exactly as required.
    We keep the dataset summary concise but informative and include a small sample.
    """
    peace = textwrap.dedent(
        f"""
        P — Purpose:
        The agent will analyze the dataset in detail as a senior data scientist.
        The dataset columns are: {column_list}

        E — Expectations:
        Provide a complete analytical pipeline including:
        - Data cleaning steps (explicit, column-by-column where useful)
        - Feature engineering strategy (what to create/transform and why)
        - Model selection rationale (and determine whether the task is regression or classification; state assumptions if unclear)
        - Full Python code for the entire ML pipeline using pandas, scikit-learn, and matplotlib (do not call any cloud services inside the code)
        - Explanation of evaluation metrics and how to interpret them (for the recommended model(s))

        A — Actions:
        1. Examine the provided dataset summary and sample.
        2. Reason about necessary transformations and possible issues.
        3. Propose the best model(s) and justify choices.
        4. Generate reproducible Python code (data loading, cleaning, feature engineering, train/test split, training, evaluation, plots).
        5. Provide clear academic-style explanations of decisions and how to interpret metrics.

        C — Constraints:
        - Do NOT invent or hallucinate columns or features. Only use columns provided in the dataset.
        - Do NOT fabricate metric values; provide code to compute metrics instead.
        - If the task type (regression vs classification) is not clear from the dataset, explicitly state that you are making an assumption and why.
        - Keep code reproducible and runnable by a graduate student with Python; include necessary imports and seed setting.
        - Use only columns present in the dataset and reference them exactly by name.
        - Keep explanations structured and organized with explicit sections.

        E — Evaluation:
        Ensure the output is:
        - Correct and logically consistent.
        - Structured with sections: Cleaning, Feature Engineering, Model Selection, Training Code, Evaluation & Interpretation.
        - Reproducible by a graduate student using Python.

        -------------------------
        Dataset summary (for the agent to analyze):
        -------------------------
        {dataset_summary_text}

        -------------------------
        Sample head (CSV, first 5 rows):
        -------------------------
        {sample_head_csv}

        -------------------------
        IMPORTANT:
        - When you generate Python code, place it inside a fenced code block labeled as python.
        - Keep explanations concise but academically rigorous.
        - If any column requires domain assumptions, state them explicitly in the Cleaning or Model Selection sections.
        """
    )
    return peace

def call_openai_peace(prompt: str, model_name: str = "gpt-4o-mini"):
    """
    Try the 'Responses' style call (new OpenAI SDK). Fallback to chat completions if needed.
    Returns text string of the LLM response.
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI package not available. Ensure 'openai' is installed.")
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        raise RuntimeError("Failed to instantiate OpenAI client. Check st.secrets['OPENAI_API_KEY'].") from e

    # Try Responses API
    try:
        resp = client.responses.create(model=model_name, input=prompt)
        # The SDK organizes output in resp.output; try to extract plain text robustly
        if hasattr(resp, "output") and resp.output:
            # Newer SDK: resp.output is list of dicts with 'content' sublists
            texts = []
            try:
                for chunk in resp.output:
                    if isinstance(chunk, dict) and "content" in chunk:
                        for item in chunk["content"]:
                            if isinstance(item, dict) and item.get("type") == "output_text":
                                texts.append(item.get("text", ""))
                if texts:
                    return "\n".join(texts)
            except Exception:
                pass
        # Fallback to stringification
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text
        # As a last resort, stringify the object
        return str(resp)
    except Exception as e_resp:
        # try Chat Completions style (older)
        try:
            chat = client.chat.completions.create(model=model_name, messages=[{"role":"user", "content": prompt}])
            # Extract text from chat response
            if hasattr(chat, "choices") and chat.choices:
                text_parts = []
                for c in chat.choices:
                    if hasattr(c, "message") and c.message:
                        text_parts.append(c.message.get("content", ""))
                    elif hasattr(c, "delta") and c.delta:
                        text_parts.append(c.delta.get("content", ""))
                return "\n".join(text_parts).strip()
            return str(chat)
        except Exception as e_chat:
            # Re-raise the original error with both tracebacks
            tb = traceback.format_exc()
            raise RuntimeError(
                "OpenAI API calls failed. Responses API error: "
                f"{e_resp}\n\nChatCompletions fallback error: {e_chat}\n\nTraceback:\n{tb}"
            ) from e_chat

# Main app flow when file uploaded
if uploaded_file is not None:
    df = safe_read_csv(uploaded_file)
    if df is None:
        st.error("Failed to read CSV — try a different encoding or a smaller file.")
    else:
        st.success("CSV loaded successfully.")
        st.header("2) Dataset preview")

        # Show head
        st.subheader("Head (first 10 rows)")
        st.dataframe(df.head(10))

        # Show df.info() as text
        st.subheader("DataFrame info()")
        info_str = df_info_to_string(df)
        st.code(info_str)

        # Rich dataset summary
        st.header("3) Dataset summary")
        with st.spinner("Computing dataset summary..."):
            summary_text = make_dataset_summary(df)
        st.text_area("Dataset summary (generated locally)", value=summary_text, height=360)

        # Quick visual: simple histograms for numeric columns (first up to 4)
        st.header("4) Quick EDA plots")
        numeric = df.select_dtypes(include=["number"])
        if numeric.shape[1] == 0:
            st.info("No numeric columns available for quick EDA plots.")
        else:
            cols_to_plot = list(numeric.columns[:4])
            for col in cols_to_plot:
                st.subheader(f"Histogram: {col}")
                fig, ax = plt.subplots()
                try:
                    df[col].dropna().hist(bins=30, ax=ax)
                    ax.set_xlabel(col)
                    ax.set_ylabel("count")
                    st.pyplot(fig)
                finally:
                    plt.close(fig)

        # Prepare prompt for LLM: limit sample head to first 5 rows
        sample_head_csv = df.head(5).to_csv(index=False)
        column_list = ", ".join(list(df.columns))

        peace_prompt = build_peace_prompt(
            dataset_summary_text=summary_text,
            column_list=column_list,
            sample_head_csv=sample_head_csv,
        )

        st.header("5) Agentic Analysis (LLM)")
        st.write(
            "The agent will analyze the dataset using the P.E.A.C.E. prompt. Click **Run agent** to send the dataset summary to the LLM."
        )

        if "OPENAI_API_KEY" not in st.secrets:
            st.warning(
                "OpenAI API key not found in st.secrets. On Render, set an environment variable OPENAI_API_KEY and add it to Streamlit secrets or use Render's built-in secrets. The app will not make LLM calls until this key is present."
            )

        run_agent = st.button("Run agent (send PEACE prompt to OpenAI)")

        if run_agent:
            if "OPENAI_API_KEY" not in st.secrets:
                st.error("Missing OpenAI API key in st.secrets['OPENAI_API_KEY']. Aborting LLM call.")
            else:
                # Show the constructed PEACE prompt in the UI (collapsible)
                with st.expander("Show constructed PEACE prompt (you can review before LLM call)"):
                    st.code(peace_prompt[:6000] + ("\n\n...truncated..." if len(peace_prompt) > 6000 else ""))

                st.info("Sending prompt to OpenAI...")
                try:
                    model_to_use = model_choice.strip() or "gpt-4o-mini"
                    llm_response = call_openai_peace(peace_prompt, model_name=model_to_use)

                    # Display results
                    st.success("Received response from LLM.")
                    if show_raw_llm:
                        st.subheader("Raw LLM output")
                        st.text_area("Raw output", value=llm_response, height=600)
                    else:
                        st.subheader("Agent Report")
                        st.markdown(
                            "Below is the agent's structured report. The agent should include sections like Cleaning, Feature Engineering, Model Selection, Code, Evaluation."
                        )
                        # Present the response in a scrollable area
                        st.text_area("Agent Report", value=llm_response, height=600)

                    # Provide download of the report
                    st.download_button(
                        label="Download full report as report.txt",
                        data=llm_response,
                        file_name="agent_report.txt",
                        mime="text/plain",
                    )

                    # Try to extract code block for convenience (simple heuristic)
                    st.subheader("Extracted Python training code (if present)")
                    # Very simple extraction: find ```python ... ``` fences
                    code_snippets = []
                    marker_start = "```python"
                    marker_end = "```"
                    idx = 0
                    while True:
                        sidx = llm_response.find(marker_start, idx)
                        if sidx == -1:
                            break
                        sidx2 = sidx + len(marker_start)
                        eidx = llm_response.find(marker_end, sidx2)
                        if eidx == -1:
                            break
                        snippet = llm_response[sidx2:eidx].strip()
                        code_snippets.append(snippet)
                        idx = eidx + len(marker_end)
                    if code_snippets:
                        for i, code in enumerate(code_snippets):
                            st.code(code, language="python")
                            st.download_button(
                                label=f"Download snippet {i+1} as code_snippet_{i+1}.py",
                                data=code,
                                file_name=f"code_snippet_{i+1}.py",
                                mime="text/x-python",
                            )
                    else:
                        st.info("No fenced python code detected in the LLM response.")

                except Exception as e:
                    st.error("LLM call failed: " + str(e))
                    st.exception(e)

else:
    st.info("Upload a CSV to begin. Small datasets (<10 MB) are recommended for quick analysis.")

# Footer / Help
st.markdown("---")
st.markdown(
    "Notes:\n"
    "- This app assembles a PEACE-structured prompt and sends a dataset summary and sample to the OpenAI API. "
    "- The model must NOT hallucinate columns; any assumptions must be explicitly stated in the agent response. "
    "- The generated code should be reviewed before running on production data."
)
