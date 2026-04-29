import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import sys
import os

# Ensure config can be imported if running directly from this folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import config
    API_URL = f"http://{config.API_HOST}:{config.API_PORT}"
except ImportError:
    API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Railway Occupancy Monitor",
    page_icon="🚉",
    layout="wide"
)

st.title("🚉 Real-Time Platform Occupancy Monitor")
st.caption("Live detection powered by YOLOv8 | Auto-refreshes every 1.5s")

# Auto-refresh
st_autorefresh(interval=1500, key="refresh")

# API call for occupancy
try:
    response = requests.get(f"{API_URL}/occupancy", timeout=1)
    response.raise_for_status()
    record = response.json()

    # Row 1 — Three metric columns
    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Person Count", record["count"])
    col2.metric("📊 Density Level", record["density"])
    col3.metric("🕐 Last Updated", record["timestamp"][-8:])

    # Density status banner
    density = record["density"]
    if density == "Low":
        st.success("✅ Platform occupancy is LOW — Normal operations")
    elif density == "Medium":
        st.warning("⚠️ Platform occupancy is MEDIUM — Monitor closely")
    elif density == "High":
        st.error("🚨 ALERT: Platform occupancy is HIGH — Take action!")
except requests.exceptions.RequestException:
    st.warning("Waiting for detection system...")

# History chart
try:
    hist_response = requests.get(f"{API_URL}/history", timeout=1)
    hist_response.raise_for_status()
    history_data = hist_response.json()

    if history_data:
        df = pd.DataFrame(history_data)
        # Select required columns
        if not df.empty:
            df = df[["timestamp", "count", "density"]]

            st.subheader("📈 Occupancy Trend (Last 50 Records)")
            st.line_chart(df.set_index("timestamp")["count"])

            # History table
            st.subheader("📋 Recent Records")
            st.dataframe(df.tail(10))
except requests.exceptions.RequestException:
    pass

# Sidebar — Configuration
st.sidebar.header("⚙️ Settings")
low_thresh = st.sidebar.slider("Low → Medium threshold", 5, 30, 15)
high_thresh = st.sidebar.slider("Medium → High threshold", 20, 80, 41)

if st.sidebar.button("Apply Thresholds"):
    try:
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
