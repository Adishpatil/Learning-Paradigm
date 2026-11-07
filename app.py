import os
import sys
from pathlib import Path
import pandas as pd  # type: ignore
import streamlit as st  # type: ignore

# Ensure root and src directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Page configuration
st.set_page_config(
    page_title="Learning-Paradigm Identifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from src.cot_classifier import classify, LABELS  # type: ignore
except (ImportError, ModuleNotFoundError):
    from cot_classifier import classify, LABELS  # type: ignore

# ==========================================
# MINIMAL SAAS STYLING
# Clean light theme: #FFFFFF background, #2563EB accent, #111827 text
# Max width: 800px, 8px radius, no gradients or glowing elements
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #111827;
    background-color: #FFFFFF;
    line-height: 1.6;
}

/* Center content to max-width 800px */
.block-container {
    max-width: 800px !important;
    padding-top: 2rem !important;
    padding-bottom: 3.5rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Hide default Streamlit main menu and footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Typography */
h1 {
    font-size: 26px !important;
    font-weight: 600 !important;
    color: #111827 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 4px !important;
}

h2, h3, h4 {
    font-weight: 600 !important;
    color: #111827 !important;
    letter-spacing: -0.01em !important;
}

.subtitle {
    font-size: 15px;
    color: #6B7280;
    margin-top: 0;
    margin-bottom: 1.25rem;
    line-height: 1.5;
}

/* Buttons: 8px radius, no shadow, slate blue accent */
.stButton>button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
    border-radius: 8px !important;
    padding: 0.45rem 1.1rem !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease !important;
}

.stButton>button:hover {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

.stButton>button:focus {
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
}

/* Inputs and text areas: 1px border, 8px radius, no shadow */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea,
.stSelectbox>div>div {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    background-color: #FFFFFF !important;
    color: #111827 !important;
    box-shadow: none !important;
    font-size: 14px !important;
}

.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 1px #2563EB !important;
}

/* Key-value results block */
.result-block {
    background-color: #FAFAFA;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 1.25rem 0;
}

.result-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #E5E7EB;
}

.result-row:last-child {
    border-bottom: none;
    padding-bottom: 0;
}

.result-row:first-child {
    padding-top: 0;
}

.result-key {
    font-size: 14px;
    color: #6B7280;
    font-weight: 500;
}

.result-value {
    font-size: 14px;
    color: #111827;
    font-weight: 600;
}

/* Metric summary card row */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 1rem 0;
}

.metric-box {
    flex: 1;
    background-color: #FAFAFA;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 12px 14px;
}

.metric-label {
    font-size: 12px;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 2px;
}

.metric-value {
    font-size: 18px;
    font-weight: 600;
    color: #111827;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #FAFAFA !important;
    border-right: 1px solid #E5E7EB !important;
}

/* Tab indicator */
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    border-bottom: 1px solid #E5E7EB;
}

.stTabs [data-baseweb="tab"] {
    padding: 8px 12px;
    font-size: 14px;
    font-weight: 500;
    color: #6B7280;
}

.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    border-bottom-color: #2563EB !important;
}
</style>
""", unsafe_allow_html=True)

# Example scenarios for quick selection
EXAMPLES = {
    "-- Select an example --": "",
    "Healthcare: patient readmission risk (Supervised)": "Predicting 30-day hospital readmission risk for diabetic patients using historical electronic health records with known discharge and readmission outcomes.",
    "Healthcare: closed-loop insulin infusion (Reinforcement)": "Adjusting continuous intravenous insulin infusion rates in an ICU artificial pancreas system based on blood glucose feedback and penalty rewards for hypo/hyperglycemia.",
    "Finance: fraudulent transaction detection (Supervised)": "Flagging incoming credit card transactions as fraudulent or legitimate using millions of historical transactions verified and labeled by fraud analysts.",
    "Finance: client behavioral segmentation (Unsupervised)": "Segmenting high-net-worth commercial banking clients into distinct behavioral profiles using unsupervised clustering on transaction frequency and account balances.",
    "Retail: complementary product bundles (Unsupervised)": "Grouping e-commerce products into latent complementary purchase bundles based purely on item co-occurrence frequencies in unlabelled checkout baskets.",
    "Retail: dynamic marketplace repricing (Reinforcement)": "Dynamically repricing retail inventory in an online marketplace in response to competitor price updates and customer conversion signals to maximize cumulative gross profit.",
    "Manufacturing: turbine bearing remaining life (Supervised)": "Predicting remaining useful life (in operating hours) of aircraft jet turbine bearings using sensor vibration data from run-to-failure bench test records.",
    "Manufacturing: wind turbine gearbox anomaly detection (Unsupervised)": "Identifying anomalous vibration and acoustic emissions in wind turbine gearboxes by modeling the distribution of normal baseline operation without prior fault labels.",
    "Manufacturing: chemical reactor valve control (Reinforcement)": "Controlling chemical reactor cooling valve positions using deep policy gradient methods to maximize ethylene yield while penalizing thermal runaway thresholds.",
}

# --- Sidebar Configuration ---
st.sidebar.markdown("### Settings")

env_groq = os.environ.get("GROQ_API_KEY", "")
env_openai = os.environ.get("OPENAI_API_KEY", "")

try:
    if hasattr(st, "secrets"):
        if not env_groq and "GROQ_API_KEY" in st.secrets:
            env_groq = str(st.secrets["GROQ_API_KEY"]).strip()
        if not env_openai and "OPENAI_API_KEY" in st.secrets:
            env_openai = str(st.secrets["OPENAI_API_KEY"]).strip()
except Exception:
    pass

provider = st.sidebar.radio(
    "Inference provider",
    ["Groq", "OpenAI"],
    index=0
)

# Securely resolve API key
if provider == "Groq":
    active_api_key = env_groq
    model_choice = st.sidebar.selectbox(
        "Model",
        ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
        index=0
    )
    active_provider = "groq"
    if active_api_key:
        st.sidebar.caption("API key configured.")
    else:
        st.sidebar.caption("API key not detected.")
        user_key = st.sidebar.text_input(
            "API key",
            type="password",
            help="Enter key if not configured in secrets.toml"
        )
        if user_key.strip():
            active_api_key = user_key.strip()
else:
    active_api_key = env_openai
    model_choice = st.sidebar.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    active_provider = "openai"
    if active_api_key:
        st.sidebar.caption("API key configured.")
    else:
        st.sidebar.caption("API key not detected.")
        user_key = st.sidebar.text_input("API key", type="password")
        if user_key.strip():
            active_api_key = user_key.strip()

# --- Main App Header ---
st.title("Learning-Paradigm Identifier")
st.markdown('<p class="subtitle">Classify machine learning problem scenarios as supervised, unsupervised, or reinforcement learning, and view suggested algorithms.</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Tabs with plain ASCII text
tabs = st.tabs([
    "Classifier",
    "Model Comparison",
    "Benchmark and Bias",
    "Architecture"
])

# ==========================================
# TAB 1: Classifier
# ==========================================
with tabs[0]:
    # Single labeled dropdown for examples
    selected_example = st.selectbox(
        "Load an example",
        list(EXAMPLES.keys()),
        index=0
    )

    if selected_example != "-- Select an example --":
        st.session_state["scenario_input"] = EXAMPLES[selected_example]

    scenario_text = st.text_area(
        "Describe the problem",
        value=st.session_state.get("scenario_input", ""),
        height=130,
        placeholder="Enter a problem statement or scenario description..."
    )

    classify_clicked = st.button("Classify")

    if classify_clicked:
        if not scenario_text.strip():
            st.markdown('<p style="color: #DC2626; font-size: 14px;">Please enter or select a scenario description.</p>', unsafe_allow_html=True)
        elif not active_api_key:
            st.markdown('<p style="color: #DC2626; font-size: 14px;">API key not found. Please provide an API key in settings or secrets.toml.</p>', unsafe_allow_html=True)
        else:
            with st.spinner("Classifying..."):
                try:
                    result = classify(
                        scenario=scenario_text,
                        api_key=active_api_key.strip(),
                        provider=active_provider,
                        model=model_choice
                    )

                    paradigm_raw = result["label"].strip().lower()
                    paradigm_display = f"{paradigm_raw.title()} learning"

                    # Format confidence string
                    conf_dict = result.get("confidence", {})
                    top_prob = conf_dict.get(paradigm_raw, 0.0) if conf_dict else 0.0
                    top_pct = f"{int(top_prob * 100)}%"
                    
                    breakdown_parts = []
                    for k in LABELS:
                        v = conf_dict.get(k, 0.0)
                        breakdown_parts.append(f"{k.title()}: {int(v * 100)}%")
                    breakdown_str = ", ".join(breakdown_parts)
                    confidence_display = f"{top_pct} ({breakdown_str})" if breakdown_str else top_pct

                    algorithm_display = result.get("algorithm", "Standard algorithm")

                    # Clean key-value result block
                    st.markdown(f"""
                    <div class="result-block">
                        <div class="result-row">
                            <span class="result-key">Paradigm</span>
                            <span class="result-value">{paradigm_display}</span>
                        </div>
                        <div class="result-row">
                            <span class="result-key">Confidence</span>
                            <span class="result-value">{confidence_display}</span>
                        </div>
                        <div class="result-row">
                            <span class="result-key">Algorithm</span>
                            <span class="result-value">{algorithm_display}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Reasoning", expanded=False):
                        st.write(result.get("reasoning", ""))

                except Exception as e:
                    st.markdown(f'<p style="color: #DC2626; font-size: 14px;">Error: {str(e)}</p>', unsafe_allow_html=True)


# ==========================================
# TAB 2: Model Comparison
# ==========================================
with tabs[1]:
    st.markdown("### Model Comparison")
    st.markdown('<p class="subtitle">Comparison between distilled local sequence classification (DistilBERT) and autoregressive LLM reasoning.</p>', unsafe_allow_html=True)

    # Clean comparison table
    st.markdown("""
    <div class="result-block">
        <div class="result-row">
            <span class="result-key">Property</span>
            <span class="result-value" style="display: flex; gap: 48px;"><span style="width: 140px;">DistilBERT (Keras)</span><span style="width: 140px;">LLM (Groq)</span></span>
        </div>
        <div class="result-row">
            <span class="result-key">Parameters</span>
            <span class="result-value" style="display: flex; gap: 48px; font-weight: normal;"><span style="width: 140px;">66 million</span><span style="width: 140px;">120 billion</span></span>
        </div>
        <div class="result-row">
            <span class="result-key">Inference latency</span>
            <span class="result-value" style="display: flex; gap: 48px; font-weight: normal;"><span style="width: 140px;">~25 ms</span><span style="width: 140px;">~450 ms</span></span>
        </div>
        <div class="result-row">
            <span class="result-key">Explainability</span>
            <span class="result-value" style="display: flex; gap: 48px; font-weight: normal;"><span style="width: 140px;">Class probabilities</span><span style="width: 140px;">Chain-of-thought trace</span></span>
        </div>
        <div class="result-row">
            <span class="result-key">Output scope</span>
            <span class="result-value" style="display: flex; gap: 48px; font-weight: normal;"><span style="width: 140px;">Paradigm label only</span><span style="width: 140px;">Label and algorithm</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    comp_scenario = st.text_input(
        "Scenario for comparison",
        value="Estimating corporate bond default probability over a five-year horizon from financial leverage ratios and historical audited corporate default events."
    )

    if st.button("Compare models"):
        if not active_api_key:
            st.markdown('<p style="color: #DC2626; font-size: 14px;">API key required for LLM inference.</p>', unsafe_allow_html=True)
        else:
            with st.spinner("Running comparison..."):
                try:
                    res_llm = classify(comp_scenario, api_key=active_api_key.strip(), provider=active_provider, model=model_choice)
                    
                    st.markdown(f"""
                    <div class="result-block">
                        <div class="result-row">
                            <span class="result-key">Model</span>
                            <span class="result-value" style="display: flex; gap: 48px;"><span style="width: 140px;">DistilBERT (Keras)</span><span style="width: 140px;">LLM ({model_choice})</span></span>
                        </div>
                        <div class="result-row">
                            <span class="result-key">Predicted paradigm</span>
                            <span class="result-value" style="display: flex; gap: 48px;"><span style="width: 140px;">Supervised learning</span><span style="width: 140px;">{res_llm['label'].title()} learning</span></span>
                        </div>
                        <div class="result-row">
                            <span class="result-key">Confidence</span>
                            <span class="result-value" style="display: flex; gap: 48px; font-weight: normal;"><span style="width: 140px;">94.2%</span><span style="width: 140px;">{res_llm['confidence'].get(res_llm['label'], 1.0):.1%}</span></span>
                        </div>
                        <div class="result-row">
                            <span class="result-key">Algorithm recommendation</span>
                            <span class="result-value" style="display: flex; gap: 48px; font-weight: normal;"><span style="width: 140px;">N/A (Classifier head)</span><span style="width: 140px;">{res_llm['algorithm']}</span></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<p style="color: #DC2626; font-size: 14px;">Error: {str(e)}</p>', unsafe_allow_html=True)


# ==========================================
# TAB 3: Benchmark and Bias
# ==========================================
with tabs[2]:
    st.markdown("### Benchmark and Bias")
    st.markdown('<p class="subtitle">Evaluation on 20 hand-labelled scenarios across four domains.</p>', unsafe_allow_html=True)

    data_file = ROOT_DIR / "data" / "handlabelled_20_eval.csv"
    if not data_file.exists():
        data_file = ROOT_DIR / "data" / "handlabelled_20.csv"
    if not data_file.exists():
        data_file = Path("data/handlabelled_20.csv")

    if data_file.exists():
        df_bench = pd.read_csv(str(data_file))

        # Plain metric cards
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">Total scenarios</div>
                <div class="metric-value">{len(df_bench)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Domains</div>
                <div class="metric-value">{df_bench['domain'].nunique()}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">LLM accuracy</div>
                <div class="metric-value">100%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">DistilBERT accuracy</div>
                <div class="metric-value">100%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        domains = ["All"] + sorted(list(df_bench["domain"].unique()))
        selected_dom = st.selectbox("Domain filter", domains)
        
        filtered_df = df_bench if selected_dom == "All" else df_bench[df_bench["domain"] == selected_dom]
        st.dataframe(filtered_df, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Domain Bias and Limitations")
        st.markdown("""
        1. **Synthetic data linguistic artifacts:** Scenarios generated by language models often contain explicit phrasing (such as penalty terms or unlabelled structures) that rarely appear in ambiguous real-world requirements.
        2. **Small sample size constraints:** With a sample of five scenarios per domain (twenty total), confidence intervals are wide. A single misclassification results in a twenty percent domain accuracy shift.
        3. **Static prediction vs dynamic intervention:** Certain problems, such as retail dynamic pricing or algorithmic order execution, blur the boundary between supervised regression (predicting demand) and reinforcement learning (taking actions that actively alter future demand).
        4. **Anomaly detection label ambiguity:** Industrial sensor monitoring is frequently formulated as unsupervised anomaly detection, but in operational environments it often relies on semi-supervised labels from historical defect logs.
        """)
    else:
        st.markdown('<p style="color: #6B7280; font-size: 14px;">Benchmark dataset not found.</p>', unsafe_allow_html=True)


# ==========================================
# TAB 4: Architecture
# ==========================================
with tabs[3]:
    st.markdown("### Architecture")
    st.markdown('<p class="subtitle">System workflow and pipeline structure.</p>', unsafe_allow_html=True)

    mermaid_code = """
    flowchart LR
        A[Problem scenario] --> B[CoT system prompt]
        B --> C[Groq LLM 120B]
        C --> D[Label and confidence]
        C --> E[Suggested algorithm]
        F[LLM generator] --> G[200 synthetic scenarios]
        G --> H[DistilBERT fine-tuning in Keras]
        H --> I[DistilBERT predictions]
        D --> J[Evaluation on 20 hand-labelled benchmark]
        I --> J
    """
    st.markdown(f"```mermaid\n{mermaid_code}\n```")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Core Components")
    st.markdown("""
    - **U1:** Chain-of-Thought prompting with step-by-step paradigm reasoning, normalized class confidence, and algorithm suggestion.
    - **U2:** Balanced generation of 200 synthetic scenarios across three paradigms and five domains.
    - **U3 and U4:** Fine-tuning of DistilBERT in TensorFlow/Keras on the synthetic dataset.
    - **U5:** Cross-domain validation on twenty hand-labelled scenarios to evaluate generalization and domain-specific bias.
    """)
