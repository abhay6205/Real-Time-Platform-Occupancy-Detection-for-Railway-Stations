"""
SYSTEM CONFIGURATION

This file acts as the master control panel for the entire Railway Occupancy System.
Decoupling these variables from the main codebase ensures that site-engineers 
can tweak performance and thresholds without needing to understand Python.
"""

# --- Video Input ---
# 0 = Default system webcam
# Or replace with a local file: "videos/test_video2.mp4" 
# Or replace with an RTSP stream: "rtsp://username:password@ip_address:port/stream"
VIDEO_SOURCE = "videos/test_videos4.mp4"

# --- Detection Model Strategy ---
# Set USE_CSRNET = True for massive crowd density estimation (Ideal for 50+ people).
# Set USE_CSRNET = False to fallback to YOLO bounding-box detection (Ideal for < 10 people).
USE_CSRNET = True

# --- CSRNet Settings ---
# Path to the massive pre-trained weights file. 
CSRNET_WEIGHTS = "models/PartAmodel_best.pth.tar"

# --- YOLO Settings (Fallback Component) ---
# These settings are only active if USE_CSRNET = False.
# MODEL_PATH = "yolov8s.pt"       # Uses the YOLOv8 small architecture for a balance of speed/accuracy
# CONFIDENCE_THRESHOLD = 0.25     # Lower confidence is needed in dense crowds due to occlusion
# NMS_IOU_THRESHOLD = 0.50        # Higher IoU allows overlapping bounding boxes (people standing close)
# INPUT_SIZE = 1280               # High resolution processing helps detect distant/small people
# PERSON_CLASS_ID = 0             # The COCO dataset class index specifically for 'person'

# --- Performance & Logic Tuning ---
# To prevent GPU overload, process 1 frame and skip the next N. 
# 3 means we process roughly 10 FPS on a 30 FPS stream, which is plenty for crowd monitoring.
FRAME_SKIP = 3                  

# Exponential Moving Average smoothing factor.
# 0.30 means the new frame only contributes 30% to the total count, smoothing out jitter.
EMA_ALPHA = 0.30                

# --- UI Threshold Classification ---
LOW_MAX = 15                    # 0 to 14 people = "Low" state (Green)
HIGH_MIN = 41                   # 41+ people = "High" state (Red)
# Note: Anything between 15 and 40 implicitly becomes the "Medium" state (Orange)

# --- Backend API Settings ---
API_HOST = "127.0.0.1"          # Host address for the FastAPI backend
API_PORT = 8000                 # Port for the FastAPI backend
DASHBOARD_REFRESH_MS = 1500     # Polling interval for legacy dashboards (1.5 seconds)

# --- Data Persistence ---
LOG_CSV_PATH = "storage/occupancy_log.csv" # Flat file Write-Ahead Log
LOG_DB_PATH  = "storage/occupancy.db"      # Relational SQLite database