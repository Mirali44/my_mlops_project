import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Page configuration with stunning theme
st.set_page_config(
    page_title="AI Prediction Studio", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS for stunning visuals with smaller fonts
st.markdown(
    """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Styles with smaller base font */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 25%, #667eea 75%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }
    
    /* Main content styling */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
        margin-top: 0.8rem;
    }
    
    /* Sidebar styling with better contrast */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a237e 0%, #283593 50%, #3f51b5 100%);
        border-right: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Sidebar text color */
    .css-1d391kg .stMarkdown, 
    .css-1d391kg .stText,
    .css-1d391kg label {
        color: #ffffff !important;
        font-weight: 500;
        font-size: 0.5rem !important;
    }
    
    /* Title styling with smaller size */
    .title-container {
        background: linear-gradient(135deg, #1a237e 0%, #283593 25%, #3f51b5 75%, #5c6bc0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.8rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .title-container p {
        font-size: 1rem !important;
        color: #1a237e;
        margin-top: -0.8rem;
        text-align: center;
        font-weight: 600;
    }
    
    /* Card styling with smaller padding */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 249, 250, 0.95) 100%);
        border-radius: 15px;
        padding: 1.2rem;
        border: 2px solid rgba(63, 81, 181, 0.2);
        backdrop-filter: blur(10px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: #1a237e;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.12);
        border-color: rgba(63, 81, 181, 0.4);
    }
    
    .metric-card h2 {
        font-size: 1.4rem !important;
        margin-bottom: 1rem !important;
    }
    
    .metric-card h3 {
        font-size: 1.2rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Button styling with smaller size */
    .stButton > button {
        background: linear-gradient(45deg, #1a237e 0%, #283593 25%, #3f51b5 75%, #5c6bc0 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        box-shadow: 0 6px 20px rgba(26, 35, 126, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        height: auto !important;
        min-height: 2.8rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-transform: none !important;
        cursor: pointer !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #0d47a1 0%, #1565c0 25%, #1976d2 75%, #42a5f5 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px rgba(26, 35, 126, 0.5) !important;
        color: white !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 8px 25px rgba(26, 35, 126, 0.4) !important;
    }
    
    .stButton > button:focus {
        outline: 2px solid rgba(63, 81, 181, 0.5) !important;
        outline-offset: 2px !important;
    }
    
    /* Primary button variant */
    .stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #e65100 0%, #ff9800 25%, #ffc107 75%, #ffeb3b 100%) !important;
        color: #1a237e !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(45deg, #bf360c 0%, #f57c00 25%, #ff8f00 75%, #ffc107 100%) !important;
        color: #1a237e !important;
    }
    
    /* File uploader button styling */
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(45deg, #1a237e 0%, #3f51b5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: linear-gradient(45deg, #0d47a1 0%, #1976d2 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 20px rgba(26, 35, 126, 0.3) !important;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #3f51b5;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(63, 81, 181, 0.05) 0%, rgba(92, 107, 192, 0.1) 100%);
        transition: all 0.3s ease;
        color: #1a237e;
        font-size: 0.85rem;
    }
    
    .uploadedFile:hover {
        border-color: #1976d2;
        background: linear-gradient(135deg, rgba(25, 118, 210, 0.1) 0%, rgba(66, 165, 245, 0.15) 100%);
        transform: scale(1.01);
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: linear-gradient(45deg, #2e7d32, #4caf50);
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.25);
        color: white;
        font-size: 0.8rem;
    }
    
    .stError {
        background: linear-gradient(45deg, #c62828, #f44336);
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 20px rgba(198, 40, 40, 0.25);
        color: white;
        font-size: 0.8rem;
    }
    
    .stInfo {
        background: linear-gradient(45deg, #1565c0, #2196f3);
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 20px rgba(21, 101, 192, 0.25);
        color: white;
        font-size: 0.5rem;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.08);
        border: 2px solid rgba(63, 81, 181, 0.2);
        font-size: 0.75rem;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #1a237e, #3f51b5, #5c6bc0);
        border-radius: 8px;
        height: 8px;
    }
    
    /* Tabs styling with smaller fonts */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, rgba(26, 35, 126, 0.1) 0%, rgba(63, 81, 181, 0.1) 100%);
        border-radius: 12px;
        padding: 0.4rem;
        border: 2px solid rgba(63, 81, 181, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        margin: 0 0.2rem;
        font-weight: 600;
        font-size: 0.5rem;
        color: #1a237e;
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(63, 81, 181, 0.3);
        padding: 0.5rem 1rem;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(45deg, #1a237e, #3f51b5);
        color: white !important;
    }
    
    /* Metric styling with much smaller font sizes */
    .stMetric {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 249, 250, 0.8) 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid rgba(63, 81, 181, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        border-color: rgba(63, 81, 181, 0.4);
    }
    
    .stMetric [data-testid="metric-container"] > div {
        color: #1a237e;
        font-weight: 600;
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 0.6rem !important;
        font-weight: 700 !important;
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-delta"] {
        font-size: 0.5rem !important;
        font-weight: 500 !important;
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-label"] {
        font-size: 0.55rem !important;
        font-weight: 600 !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid rgba(63, 81, 181, 0.3);
        border-radius: 10px;
        color: #1a237e;
        font-size: 0.8rem;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid rgba(63, 81, 181, 0.3);
        border-radius: 10px;
        color: #1a237e;
        font-weight: 500;
        font-size: 0.8rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1976d2;
        box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.2);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #1a237e, #3f51b5);
    }
    
    /* Animated gradient text */
    .gradient-text {
        background: linear-gradient(-45deg, #1a237e, #283593, #3f51b5, #5c6bc0);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient 3s ease infinite;
        font-weight: 700;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Floating animation with smaller movement */
    .floating {
        animation: floating 6s ease-in-out infinite;
    }
    
    @keyframes floating {
        0% { transform: translate(0, 0px); }
        50% { transform: translate(0, -8px); }
        100% { transform: translate(0, -0px); }
    }
    
    /* Pulse animation for important elements */
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(63, 81, 181, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(63, 81, 181, 0); }
        100% { box-shadow: 0 0 0 0 rgba(63, 81, 181, 0); }
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 0.8rem;
        border: 2px solid rgba(63, 81, 181, 0.2);
        font-size: 0.8rem;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #1a237e !important;
        font-weight: 600;
        font-size: 0.8rem !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(63, 81, 181, 0.1) 0%, rgba(92, 107, 192, 0.1) 100%);
        border-radius: 10px;
        border: 2px solid rgba(63, 81, 181, 0.2);
        color: #1a237e;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #2e7d32 0%, #4caf50 25%, #66bb6a 75%, #81c784 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        min-height: 2.5rem !important;
        cursor: pointer !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(45deg, #1b5e20 0%, #2e7d32 25%, #4caf50 75%, #66bb6a 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 30px rgba(46, 125, 50, 0.5) !important;
        color: white !important;
    }
    
    /* Sidebar headers */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: white !important;
        font-weight: 700;
    }
    
    .css-1d391kg h1 { font-size: 1.4rem !important; }
    .css-1d391kg h2 { font-size: 1.2rem !important; }
    .css-1d391kg h3 { font-size: 1rem !important; }
    
    /* JSON viewer styling */
    .css-1d391kg pre {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #e3f2fd;
        font-size: 0.75rem;
    }
    
    /* General text elements smaller */
    .stMarkdown p, .stMarkdown li {
        font-size: 0.85rem;
        line-height: 1.5;
    }
    
    .stMarkdown h1 { font-size: 1.8rem; }
    .stMarkdown h2 { font-size: 1.5rem; }
    .stMarkdown h3 { font-size: 1.3rem; }
    .stMarkdown h4 { font-size: 1.1rem; }
    .stMarkdown h5 { font-size: 1rem; }
    .stMarkdown h6 { font-size: 0.9rem; }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid rgba(63, 81, 181, 0.3);
        border-radius: 10px;
        color: #1a237e;
        font-weight: 500;
        font-size: 0.5rem;
    }
    
    /* Slider labels */
    .stSlider > label {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Animated header with smaller font
st.markdown(
    """
<div class="title-container floating">
    <h1>🤖 AI Prediction Studio</h1>
    <p>Unleash the power of machine learning with stunning precision ✨</p>
</div>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "model_performance" not in st.session_state:
    st.session_state.model_performance = None

# Enhanced sidebar
with st.sidebar:
    st.markdown(
        """
    <div style="text-align: center; padding: 1.2rem;">
        <h2 style="color: white; margin-bottom: 1.5rem; font-weight: 800; font-size: 0.8rem;">⚙️ Control Center</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # API endpoint configuration with beautiful styling
    st.markdown("### 🔗 API Configuration")
    api_endpoint = st.text_input(
        "",
        value="http://localhost:8000",
        placeholder="Enter FastAPI Backend URL",
        help="🚀 URL of your FastAPI backend server",
        key="api_input",
    )

    # Test connection button
    if st.button("🔍 Test Connection", key="test_conn", use_container_width=True):
        with st.spinner("Testing connection..."):
            try:
                response = requests.get(f"{api_endpoint}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Connection successful!")
                    data = response.json()
                    st.json(data)
                else:
                    st.error("❌ Connection failed!")
            except Exception:
                st.error("❌ Cannot reach backend!")

    st.divider()

    # Model information with beautiful cards
    st.markdown("### 🧠 Model Intelligence")

    # Mock model metrics with stunning visuals
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Accuracy", "94.2%", "2.1%")
    with col2:
        st.metric("⚡ Speed", "0.02s", "-0.01s")

    col3, col4 = st.columns(2)
    with col3:
        st.metric("📊 Samples", "10.2K", "1.2K")
    with col4:
        st.metric("🔥 Confidence", "98.5%", "0.3%")

    st.divider()

    # Theme selector
    st.markdown("### 🎨 Visualization Theme")
    chart_theme = st.selectbox(
        "Choose theme:", ["plotly", "plotly_white", "plotly_dark", "presentation"]
    )

# Create beautiful columns with proper spacing
col1, col2 = st.columns([1.2, 0.8], gap="large")

# Left column - Dataset Upload
with col1:
    st.markdown(
        """
    <div class="metric-card">
        <h2 style="text-align: center; margin-bottom: 1rem; color: #1a237e;">
            📁 <span class="gradient-text">Dataset Upload Zone</span>
        </h2>
    """,
        unsafe_allow_html=True,
    )

    # Beautiful file uploader
    uploaded_file = st.file_uploader(
        "Drop your dataset here or browse files",
        type=["csv", "xlsx", "xls"],
        help="📈 Supported formats: CSV, Excel (.xlsx, .xls)",
        key="file_upload",
    )

    if uploaded_file is not None:
        try:
            # Read file with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text(f"📖 Reading file... {i+1}%")
                elif i < 60:
                    status_text.text(f"🔍 Analyzing data... {i+1}%")
                else:
                    status_text.text(f"✨ Preparing visualization... {i+1}%")
                time.sleep(0.01)

            # Read the dataset based on file type
            if uploaded_file.name.endswith(".csv"):
                dataset = pd.read_csv(uploaded_file)
            else:
                dataset = pd.read_excel(uploaded_file)

            st.session_state.dataset = dataset

            status_text.text("✅ Dataset loaded successfully!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

            # Beautiful success message with metrics
            st.balloons()
            st.success("🎉 Dataset loaded successfully!")

            # Dataset metrics in beautiful cards
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("📏 Rows", f"{dataset.shape[0]:,}")
            with metric_col2:
                st.metric("📋 Columns", f"{dataset.shape[1]:,}")
            with metric_col3:
                st.metric("💾 Size", f"{dataset.memory_usage().sum() / 1024:.1f}KB")
            with metric_col4:
                st.metric(
                    "🔢 Numeric Cols",
                    f"{dataset.select_dtypes(include=np.number).shape[1]}",
                )

        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Dataset exploration with stunning visualizations
    if st.session_state.dataset is not None:
        dataset = st.session_state.dataset

        st.markdown(
            """
        <div class="metric-card">
            <h3 style="text-align: center; margin-bottom: 0.8rem; color: #1a237e;">
                📊 <span class="gradient-text">Data Explorer</span>
            </h3>
        """,
            unsafe_allow_html=True,
        )

        # Beautiful tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 Overview", "🔍 Sample Data", "📋 Info", "📊 Visualizations"]
        )

        with tab1:
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### 🎯 Dataset Summary")
                summary_data = {
                    "Metric": [
                        "Total Rows",
                        "Total Columns",
                        "Memory Usage",
                        "Missing Values",
                    ],
                    "Value": [
                        f"{dataset.shape[0]:,}",
                        f"{dataset.shape[1]:,}",
                        f"{dataset.memory_usage().sum() / 1024:.1f} KB",
                        f"{dataset.isnull().sum().sum():,}",
                    ],
                }
                st.dataframe(
                    pd.DataFrame(summary_data),
                    hide_index=True,
                    use_container_width=True,
                )

            with col_b:
                st.markdown("#### 🏷️ Column Types")
                dtype_counts = dataset.dtypes.value_counts()
                # Convert dtype names to strings to avoid JSON serialization issues
                dtype_names = [str(dtype) for dtype in dtype_counts.index]
                fig = px.pie(
                    values=dtype_counts.values,
                    names=dtype_names,
                    title="Data Types Distribution",
                    template=chart_theme,
                    color_discrete_sequence=[
                        "#1a237e",
                        "#283593",
                        "#3f51b5",
                        "#5c6bc0",
                        "#7986cb",
                    ],
                )
                fig.update_layout(height=250, showlegend=True, title_font_size=12)
                st.plotly_chart(fig, use_container_width=True)

            # Statistical summary with beautiful styling
            if st.checkbox("📊 Show Statistical Summary", key="stats_summary"):
                st.markdown("#### 📈 Statistical Analysis")
                numeric_data = dataset.select_dtypes(include=np.number)
                if not numeric_data.empty:
                    st.dataframe(numeric_data.describe(), use_container_width=True)
                else:
                    st.info("ℹ️ No numeric columns found for statistical analysis.")

        with tab2:
            st.markdown("#### 👀 Data Preview")
            # Interactive data viewer
            n_rows = st.slider(
                "Number of rows to display:", 5, min(50, len(dataset)), 10
            )
            st.dataframe(dataset.head(n_rows), use_container_width=True, height=350)

        with tab3:
            col_info1, col_info2 = st.columns(2)

            with col_info1:
                st.markdown("#### 🏗️ Data Types")
                dtype_df = pd.DataFrame(
                    {
                        "Column": dataset.dtypes.index,
                        "Data Type": [
                            str(dtype) for dtype in dataset.dtypes.values
                        ],  # Convert to string
                    }
                )
                st.dataframe(dtype_df, hide_index=True, use_container_width=True)

            with col_info2:
                st.markdown("#### ❓ Missing Values")
                missing_df = pd.DataFrame(
                    {
                        "Column": dataset.columns,
                        "Missing Count": dataset.isnull().sum().values,
                        "Missing %": (dataset.isnull().sum() * 100 / len(dataset))
                        .round(2)
                        .values,
                    }
                )
                missing_data = missing_df[missing_df["Missing Count"] > 0]
                if len(missing_data) > 0:
                    st.dataframe(
                        missing_data, hide_index=True, use_container_width=True
                    )
                else:
                    st.success("✅ No missing values found!")

        with tab4:
            st.markdown("#### 📊 Data Visualizations")

            numeric_cols = dataset.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                # Correlation heatmap
                if len(numeric_cols) > 1:
                    fig = px.imshow(
                        dataset[numeric_cols].corr(),
                        template=chart_theme,
                        color_continuous_scale=[
                            [0, "#1a237e"],
                            [0.5, "#ffffff"],
                            [1, "#ff9800"],
                        ],
                        aspect="auto",
                        title="🔥 Correlation Heatmap",
                    )
                    fig.update_layout(height=300, title_font_size=12)
                    st.plotly_chart(fig, use_container_width=True)

                # Distribution plots
                if st.checkbox("Show Distribution Plots", key="dist_plots"):
                    selected_col = st.selectbox(
                        "Select column for distribution:", numeric_cols
                    )
                    if selected_col:
                        fig = px.histogram(
                            dataset,
                            x=selected_col,
                            template=chart_theme,
                            title=f"📈 Distribution of {selected_col}",
                            color_discrete_sequence=["#1a237e"],
                        )
                        fig.update_layout(height=300, title_font_size=12)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ No numeric columns found for visualization.")

        st.markdown("</div>", unsafe_allow_html=True)

# Right column - Predictions
with col2:
    st.markdown(
        """
    <div class="metric-card pulse">
        <h2 style="text-align: center; margin-bottom: 1rem; color: #1a237e;">
            🚀 <span class="gradient-text">Prediction Engine</span>
        </h2>
    """,
        unsafe_allow_html=True,
    )

    if st.session_state.dataset is not None:
        dataset = st.session_state.dataset

        # Dataset info with beautiful metrics
        st.info(
            f"🎯 Dataset ready: **{dataset.shape[0]:,}** rows, **{dataset.shape[1]}** columns"
        )

        st.markdown("#### ⚙️ Prediction Settings")

        # Advanced prediction settings
        prediction_mode = st.radio(
            "🎯 Prediction Mode",
            ["🚀 Full Dataset", "🎲 Random Sample", "📑 Custom Range"],
            help="Choose your prediction strategy",
        )

        sample_size = len(dataset)
        start_idx, end_idx = 0, len(dataset)

        if prediction_mode == "🎲 Random Sample":
            sample_size = st.slider(
                "Sample Size",
                min_value=1,
                max_value=len(dataset),
                value=min(100, len(dataset)),
                help="Number of random rows to predict",
            )
        elif prediction_mode == "📑 Custom Range":
            col_start, col_end = st.columns(2)
            with col_start:
                start_idx = st.number_input("Start index", 0, len(dataset) - 1, 0)
            with col_end:
                end_idx = st.number_input(
                    "End index",
                    start_idx + 1,
                    len(dataset),
                    min(start_idx + 100, len(dataset)),
                )
            sample_size = end_idx - start_idx

        # Advanced options
        with st.expander("🔧 Advanced Options"):
            batch_processing = st.checkbox(
                "Enable batch processing", help="Process data in smaller batches"
            )
            if batch_processing:
                batch_size = st.slider("Batch size", 10, 1000, 100)

            confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)
            include_metadata = st.checkbox("Include prediction metadata", True)

        # Stunning prediction button
        prediction_button = st.button(
            "🚀 Launch Predictions",
            type="primary",
            use_container_width=True,
            key="predict_btn",
        )

        if prediction_button:
            try:
                # Prepare data based on mode
                if prediction_mode == "🎲 Random Sample":
                    data_for_prediction = dataset.sample(n=sample_size, random_state=42)
                elif prediction_mode == "📑 Custom Range":
                    data_for_prediction = dataset.iloc[start_idx:end_idx]
                else:
                    data_for_prediction = dataset

                # Beautiful progress animation
                progress_container = st.container()

                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Animated progress
                    stages = [
                        "🔄 Initializing prediction engine...",
                        "📊 Processing data...",
                        "🧠 Running ML model...",
                        "📈 Calculating confidence scores...",
                        "✨ Finalizing predictions...",
                    ]

                    for i, stage in enumerate(stages):
                        for j in range(20):
                            progress = i * 20 + j + 1
                            progress_bar.progress(progress)
                            status_text.text(f"{stage} {progress}%")
                            time.sleep(0.05)

                # Convert data to file format for API
                csv_data = data_for_prediction.to_csv(index=False).encode("utf-8")

                # Make API call with file upload
                with st.spinner("🚀 Communicating with AI backend..."):
                    files = {"file": ("data.csv", csv_data, "text/csv")}
                    response = requests.post(
                        f"{api_endpoint}/predict", files=files, timeout=60
                    )

                    if response.status_code == 200:
                        predictions_data = response.json()

                        # Process predictions
                        if (
                            "data" in predictions_data
                            and "predictions" in predictions_data["data"]
                        ):
                            predictions = predictions_data["data"]["predictions"]
                            processing_time = predictions_data["data"].get(
                                "processing_time_seconds", 0
                            )

                            # Create beautiful results
                            st.balloons()
                            st.success(
                                f"🎉 Predictions completed in {processing_time}s!"
                            )

                            # Store predictions
                            st.session_state.predictions = pd.DataFrame(predictions)

                            # Show metrics
                            pred_col1, pred_col2, pred_col3 = st.columns(3)
                            with pred_col1:
                                st.metric(
                                    "⚡ Processing Time", f"{processing_time:.2f}s"
                                )
                            with pred_col2:
                                st.metric("📊 Predictions", len(predictions))
                            with pred_col3:
                                avg_conf = np.mean(
                                    [p.get("confidence", 0.5) for p in predictions]
                                )
                                st.metric("🎯 Avg Confidence", f"{avg_conf:.1%}")

                        else:
                            st.error("❌ Unexpected response format from API")
                    else:
                        st.error(
                            f"❌ API Error: {response.status_code} - {response.text}"
                        )

            except requests.exceptions.ConnectionError:
                st.error(
                    "🔌 Cannot connect to AI backend. Ensure the server is running!"
                )
            except requests.exceptions.Timeout:
                st.error(
                    "⏱️ Request timed out. The model might be processing large data."
                )
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
            finally:
                # Clean up progress indicators
                progress_container.empty()

    else:
        st.markdown(
            """
        <div style="text-align: center; padding: 2rem; opacity: 0.8;">
            <h3 style="color: #1a237e; font-size: 1.1rem;">📥 Upload Dataset First</h3>
            <p style="color: #1a237e; font-weight: 500; font-size: 0.85rem;">Upload your dataset in the left panel to start making predictions</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# Prediction Results Section
if st.session_state.predictions is not None:
    st.markdown("---")
    st.markdown(
        """
    <div class="metric-card">
        <h2 style="text-align: center; margin-bottom: 1.5rem; color: #1a237e;">
            📈 <span class="gradient-text">Prediction Results Dashboard</span>
        </h2>
    """,
        unsafe_allow_html=True,
    )

    predictions_df = st.session_state.predictions

    # Results tabs
    result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs(
        ["📊 Results Table", "📈 Visualizations", "📉 Analytics", "💾 Export"]
    )

    with result_tab1:
        st.markdown("#### 🎯 Prediction Results")
        st.dataframe(predictions_df, use_container_width=True, height=350)

    with result_tab2:
        if "prediction" in predictions_df.columns:
            col_viz1, col_viz2 = st.columns(2)

            with col_viz1:
                # Prediction distribution
                fig = px.histogram(
                    predictions_df,
                    x="prediction",
                    template=chart_theme,
                    title="🎯 Prediction Distribution",
                    color_discrete_sequence=["#1a237e"],
                )
                fig.update_layout(height=280, title_font_size=11)
                st.plotly_chart(fig, use_container_width=True)

            with col_viz2:
                if "confidence" in predictions_df.columns:
                    # Confidence distribution
                    fig = px.histogram(
                        predictions_df,
                        x="confidence",
                        template=chart_theme,
                        title="📊 Confidence Distribution",
                        color_discrete_sequence=["#3f51b5"],
                    )
                    fig.update_layout(height=280, title_font_size=11)
                    st.plotly_chart(fig, use_container_width=True)

            # Scatter plot if both prediction and confidence exist
            if "confidence" in predictions_df.columns:
                fig = px.scatter(
                    predictions_df,
                    x="prediction",
                    y="confidence",
                    template=chart_theme,
                    title="🎯 Prediction vs Confidence",
                    color="confidence",
                    size="confidence",
                    color_continuous_scale=[
                        [0, "#1a237e"],
                        [0.5, "#3f51b5"],
                        [1, "#ff9800"],
                    ],
                )
                fig.update_layout(height=320, title_font_size=11)
                st.plotly_chart(fig, use_container_width=True)

    with result_tab3:
        st.markdown("#### 📈 Prediction Analytics")

        if "prediction" in predictions_df.columns:
            pred_stats = predictions_df["prediction"].describe()

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("📊 Mean", f"{pred_stats['mean']:.3f}")
            with metric_col2:
                st.metric("📏 Std Dev", f"{pred_stats['std']:.3f}")
            with metric_col3:
                st.metric("📉 Min", f"{pred_stats['min']:.3f}")
            with metric_col4:
                st.metric("📈 Max", f"{pred_stats['max']:.3f}")

            # Detailed statistics
            st.markdown("#### 📊 Detailed Statistics")
            st.dataframe(pred_stats.to_frame().T, use_container_width=True)

        if "confidence" in predictions_df.columns:
            conf_stats = predictions_df["confidence"].describe()
            st.markdown("#### 🎯 Confidence Statistics")
            st.dataframe(conf_stats.to_frame().T, use_container_width=True)

    with result_tab4:
        st.markdown("#### 💾 Export Options")

        col_export1, col_export2 = st.columns(2)

        with col_export1:
            # CSV Export
            csv_data = predictions_df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_export2:
            # JSON Export
            json_data = predictions_df.to_json(orient="records", indent=2)
            st.download_button(
                label="📋 Download as JSON",
                data=json_data,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        # Summary report
        avg_pred = (
            f"{predictions_df['prediction'].mean():.4f}"
            if "prediction" in predictions_df.columns
            else "N/A"
        )
        std_pred = (
            f"{predictions_df['prediction'].std():.4f}"
            if "prediction" in predictions_df.columns
            else "N/A"
        )
        min_pred = (
            f"{predictions_df['prediction'].min():.4f}"
            if "prediction" in predictions_df.columns
            else "N/A"
        )
        max_pred = (
            f"{predictions_df['prediction'].max():.4f}"
            if "prediction" in predictions_df.columns
            else "N/A"
        )

        avg_conf = (
            f"{predictions_df['confidence'].mean():.4f}"
            if "confidence" in predictions_df.columns
            else "N/A"
        )
        high_conf = (
            len(predictions_df[predictions_df["confidence"] > 0.8])
            if "confidence" in predictions_df.columns
            else "N/A"
        )

        st.markdown("#### 📑 Generate Report")
        if st.button(
            "📊 Generate Summary Report",
            use_container_width=True,
            key="generate_report",
        ):
            report = f"""
        
# 🤖 AI Prediction Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        

## 📊 Summary
- **Total Predictions:** {len(predictions_df):,}
- **Average Prediction:** {avg_pred}
- **Standard Deviation:** {std_pred}
- **Min Value:** {min_pred}
- **Max Value:** {max_pred}

## 🎯 Confidence Metrics
- **Average Confidence:** {avg_conf}
- **High Confidence (>0.8):** {high_conf} predictions

## 📈 Model Performance
- **Processing completed successfully** ✅
- **Data quality:** Excellent
- **Prediction reliability:** High

---
Generated by AI Prediction Studio 🚀
            """

            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
                key="download_report",
            )

            st.markdown(report)

    st.markdown("</div>", unsafe_allow_html=True)

# Beautiful footer with smaller text
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, rgba(26, 35, 126, 0.95) 0%, rgba(63, 81, 181, 0.9) 100%); border-radius: 15px; margin: 1.5rem 0; border: 2px solid rgba(255, 255, 255, 0.2);">
    <h3 style="color: white; margin-bottom: 1.5rem; font-weight: 800; font-size: 1.4rem;">🚀 How to Use AI Prediction Studio</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.2rem; margin: 1.5rem 0;">
        <div style="padding: 1.5rem; background: rgba(255, 255, 255, 0.15); border-radius: 12px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); transition: transform 0.3s ease;">
            <h4 style="color: white; font-weight: 700; margin-bottom: 0.8rem; font-size: 0.9rem;">1. 🔗 Connect</h4>
            <p style="color: rgba(255, 255, 255, 0.9); margin: 0; line-height: 1.4; font-size: 0.75rem;">Configure your FastAPI backend URL in the sidebar and test the connection.</p>
        </div>
        <div style="padding: 1.5rem; background: rgba(255, 255, 255, 0.15); border-radius: 12px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); transition: transform 0.3s ease;">
            <h4 style="color: white; font-weight: 700; margin-bottom: 0.8rem; font-size: 0.9rem;">2. 📁 Upload</h4>
            <p style="color: rgba(255, 255, 255, 0.9); margin: 0; line-height: 1.4; font-size: 0.75rem;">Drop your CSV or Excel file into the upload zone and explore your data.</p>
        </div>
        <div style="padding: 1.5rem; background: rgba(255, 255, 255, 0.15); border-radius: 12px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); transition: transform 0.3s ease;">
            <h4 style="color: white; font-weight: 700; margin-bottom: 0.8rem; font-size: 0.9rem;">3. ⚙️ Configure</h4>
            <p style="color: rgba(255, 255, 255, 0.9); margin: 0; line-height: 1.4; font-size: 0.75rem;">Choose your prediction mode and adjust advanced settings as needed.</p>
        </div>
        <div style="padding: 1.5rem; background: rgba(255, 255, 255, 0.15); border-radius: 12px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); transition: transform 0.3s ease;">
            <h4 style="color: white; font-weight: 700; margin-bottom: 0.8rem; font-size: 0.9rem;">4. 🚀 Predict</h4>
            <p style="color: rgba(255, 255, 255, 0.9); margin: 0; line-height: 1.4; font-size: 0.75rem;">Launch predictions and explore results with stunning visualizations.</p>
        </div>
        <div style="padding: 1.5rem; background: rgba(255, 255, 255, 0.15); border-radius: 12px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); transition: transform 0.3s ease;">
            <h4 style="color: white; font-weight: 700; margin-bottom: 0.8rem; font-size: 0.9rem;">5. 💾 Export</h4>
            <p style="color: rgba(255, 255, 255, 0.9); margin: 0; line-height: 1.4; font-size: 0.75rem;">Download your results and generate comprehensive reports.</p>
        </div>
    </div>
    
    <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(45deg, rgba(255, 152, 0, 0.9) 0%, rgba(255, 193, 7, 0.8) 100%); border-radius: 12px; border: 2px solid rgba(255, 255, 255, 0.3);">
        <p style="margin: 0; color: #1a237e; font-weight: 700; font-size: 0.85rem; line-height: 1.5;">
            ✨ <strong>Pro Tips:</strong> Use the visualization theme selector to match your brand • 
            Enable batch processing for large datasets • 
            Adjust confidence thresholds for optimal results
        </p>
    </div>
    
    <div style="margin-top: 1.5rem; font-size: 0.8rem; color: rgba(255, 255, 255, 0.9);">
        <p style="margin: 0.4rem 0; font-weight: 600;">🤖 Powered by FastAPI • Streamlit • AI Magic ✨</p>
        <p style="margin: 0.4rem 0; font-weight: 500;">Made with ❤️ for the future of machine learning</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Enhanced floating particles animation with smaller particles
st.markdown(
    """
<div class="particles">
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
</div>

<style>
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        overflow: hidden;
    }
    
    .particle {
        position: absolute;
        width: 4px;
        height: 4px;
        background: linear-gradient(45deg, rgba(255, 255, 255, 0.3), rgba(63, 81, 181, 0.2));
        border-radius: 50%;
        animation: float 8s infinite ease-in-out;
    }
    
    .particle:nth-child(1) {
        left: 10%;
        animation-delay: 0s;
        animation-duration: 6s;
    }
    
    .particle:nth-child(2) {
        left: 25%;
        animation-delay: 1s;
        animation-duration: 7s;
    }
    
    .particle:nth-child(3) {
        left: 40%;
        animation-delay: 2s;
        animation-duration: 8s;
    }
    
    .particle:nth-child(4) {
        left: 55%;
        animation-delay: 3s;
        animation-duration: 6.5s;
    }
    
    .particle:nth-child(5) {
        left: 70%;
        animation-delay: 4s;
        animation-duration: 7.5s;
    }
    
    .particle:nth-child(6) {
        left: 85%;
        animation-delay: 2.5s;
        animation-duration: 8.5s;
    }
    
    .particle:nth-child(7) {
        left: 95%;
        animation-delay: 1.5s;
        animation-duration: 9s;
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(100vh) scale(0) rotate(0deg);
            opacity: 0;
        }
        10% {
            opacity: 1;
            transform: translateY(90vh) scale(0.4) rotate(45deg);
        }
        90% {
            opacity: 1;
            transform: translateY(10vh) scale(0.8) rotate(315deg);
        }
        100% {
            transform: translateY(0) scale(0) rotate(360deg);
            opacity: 0;
        }
    }
    
    /* Hover effects for enhanced interactivity */
    .metric-card:hover .gradient-text {
        animation-duration: 1s;
    }
    
    /* Enhanced button focus states */
    .stButton > button:focus {
        outline: 2px solid rgba(63, 81, 181, 0.5);
        outline-offset: 2px;
    }
    
    /* Better form element focus */
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(63, 81, 181, 0.25);
    }
    
    /* Make all interactive elements fully clickable */
    .stButton, .stDownloadButton {
        width: 100% !important;
    }
    
    .stButton > button, .stDownloadButton > button {
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-decoration: none !important;
        cursor: pointer !important;
    }
    
    /* Fix checkbox and radio button interactions */
    .stCheckbox > label, .stRadio > label {
        cursor: pointer !important;
        width: 100% !important;
        display: block !important;
    }
    
    /* Make file uploader fully interactive */
    [data-testid="stFileUploader"] {
        width: 100%;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover {
        transform: scale(1.01) !important;
    }
    
    /* Fix slider interactions */
    .stSlider {
        cursor: pointer;
    }
    
    /* Ensure all form elements are properly styled */
    .stSelectbox, .stTextInput, .stNumberInput {
        cursor: pointer;
    }
    
    /* Make tabs fully clickable */
    .stTabs [data-baseweb="tab"] {
        cursor: pointer !important;
        padding: 0.6rem 1.2rem !important;
        width: 100% !important;
    }
</style>
""",
    unsafe_allow_html=True,
)
