import streamlit as st  # PURPOSE: Streamlit framework - creates interactive web dashboard UI without HTML/JavaScript/CSS coding
from streamlit_autorefresh import st_autorefresh  # PURPOSE: Auto-refresh plugin - automatically polls API at regular intervals to fetch live data
import requests  # PURPOSE: HTTP client library - makes GET/POST requests to backend FastAPI to fetch occupancy records and video stream
import pandas as pd  # PURPOSE: Data analysis library - converts API history into DataFrames for chart visualization and tabular display
import sys  # PURPOSE: System utilities - manipulates Python path to import config module from parent directory
import os  # PURPOSE: File system operations - constructs file paths for module imports

# Ensure config can be imported if running directly from this folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import config  # PURPOSE: Configuration module - reads API_HOST and API_PORT settings
    API_URL = f"http://{config.API_HOST}:{config.API_PORT}"
except ImportError:
    API_URL = "http://127.0.0.1:8000"

# Streamlit: Configures page metadata and layout for web dashboard
st.set_page_config(
    page_title="Railway Occupancy Monitor",
    page_icon="🚉",
    layout="wide"
)

st.title("🚉 Real-Time Platform Occupancy Monitor")
st.caption("Live detection powered by YOLOv8 | Auto-refreshes every 1.5s")

# streamlit_autorefresh: Auto-refreshes the entire page every 1500ms (1.5 seconds)
# Fetches latest data from API without requiring manual page reload
st_autorefresh(interval=1500, key="refresh")

# requests: Makes HTTP GET request to backend API to fetch current occupancy record
# timeout=1: Waits max 1 second for API response before giving up
try:
    response = requests.get(f"{API_URL}/occupancy", timeout=1)
    response.raise_for_status()
    record = response.json()

    # Streamlit: Creates three metric cards in a row displaying current state
    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Person Count", record["count"])  # Uses Pydantic model to display count
    col2.metric("📊 Density Level", record["density"])
    col3.metric("🕐 Last Updated", record["timestamp"][-8:])

    # Streamlit: Displays color-coded status banner based on density classification
    # OccupancyRecord.density: Comes from DensityClassifier output (Low/Medium/High)
    density = record["density"]
    if density == "Low":
        st.success("✅ Platform occupancy is LOW — Normal operations")
    elif density == "Medium":
        st.warning("⚠️ Platform occupancy is MEDIUM — Monitor closely")
    elif density == "High":
        st.error("🚨 ALERT: Platform occupancy is HIGH — Take action!")
except requests.exceptions.RequestException:
    st.warning("Waiting for detection system...")

# requests: Makes GET request to fetch last 50 occupancy records for historical analysis
try:
    hist_response = requests.get(f"{API_URL}/history", timeout=1)
    hist_response.raise_for_status()
    history_data = hist_response.json()

    if history_data:
        # pandas: Converts JSON history list into DataFrame for visualization and manipulation
        df = pd.DataFrame(history_data)
        # Select required columns
        if not df.empty:
            df = df[["timestamp", "count", "density"]]

            st.subheader("📈 Occupancy Trend (Last 50 Records)")
            # Streamlit line chart: Automatically plots count over time from pandas DataFrame
            st.line_chart(df.set_index("timestamp")["count"])

            # Streamlit dataframe: Displays last 10 records in interactive sortable table
            st.subheader("📋 Recent Records")
            st.dataframe(df.tail(10))
except requests.exceptions.RequestException:
    pass

# Streamlit: Creates collapsible sidebar for configuration options
st.sidebar.header("⚙️ Settings")
low_thresh = st.sidebar.slider("Low → Medium threshold", 5, 30, 15)
high_thresh = st.sidebar.slider("Medium → High threshold", 20, 80, 41)

if st.sidebar.button("Apply Thresholds"):
    try:
        # requests: Makes POST request to update density classification thresholds dynamically
        # No need to restart system; API updates classifier in real-time
        res = requests.post(
            f"{API_URL}/thresholds",
            json={"low_max": low_thresh, "high_min": high_thresh},
            timeout=1
        )
        res.raise_for_status()
        st.sidebar.success("Thresholds updated!")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Failed to update thresholds: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Green = Low | Orange = Medium | Red = High")
