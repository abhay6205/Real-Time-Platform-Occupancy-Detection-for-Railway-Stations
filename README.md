# Real-Time Platform Occupancy Detection for Railway Stations

## Overview
This system provides real-time detection of passenger occupancy on high-density railway platforms. Because typical object-detection models struggle when faced with hundreds of heavily occluded people, this project utilizes **CSRNet (Dilated Convolutional Neural Networks)** for accurate Crowd Density Estimation.

It processes video feeds, computes density heatmaps, logs the data to a SQLite database, and features a FastAPI backend combined with a Vite/React dashboard for real-time monitoring. 

*(Note: YOLOv8 configuration fallback is preserved for sparse-environment monitoring).*

## Project Structure
We have structured the application components into clearly separated modules to ensure high cohesiveness and loose coupling.

```text
├── backend/                   # Core Python application (FastAPI + AI model)
│   ├── api/                   # FastAPI routing and models
│   ├── counter/               # Estimation logic (Person Counter, Density Classifier)
│   ├── detector/              # Models and processing logic (CSRNet, YOLO)
│   ├── models/                # Saved weights for ML models
│   ├── storage/               # Logging and SQLite database logic
│   ├── tests/                 # Pytest suite
│   ├── videos/                # Sample input streams/videos
│   ├── config.py              # Central configuration module
│   ├── Dockerfile             # Container definition for the backend
│   ├── main.py                # Pipeline orchestrator
│   └── requirements.txt       # Python dependencies
├── frontend/                  # React + Vite Web Dashboard
│   ├── src/                   # React source code and styling
│   ├── package.json           # Node.js dependencies
│   └── vite.config.js         # Vite configuration
└── docs/                      # Related documentation and project reports
```

## Setup & Installation

### 1. Backend API (Python)
Navigate to the `backend/` directory:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```
To run the pipeline and backend API server:
```bash
python main.py
```
*(The backend defaults to running locally on port 8000).*

### 2. Frontend Dashboard (Node)
Navigate to the `frontend/` directory:
```bash
cd frontend
npm install
npm run dev
```
*(The frontend defaults to running locally via Vite, generally on port 5173).*

### 3. Docker (Optional)
The backend includes a `Dockerfile` for easy deployment:
```bash
cd backend
docker build -t railway-occupancy-backend .
docker run -p 8000:8000 railway-occupancy-backend
```

## Features
- **Real-Time Video Capture**: Connect streams such as RTSP, Webcams, or MP4 files transparently.
- **CSRNet Crowding Analytics**: Deep learning analysis optimized for mass crowds.
- **FastAPI Backend**: Rapid, decoupled web data delivery and logging to SQLite.
- **Interactive UI**: A React-based web dashboard reflecting platform saturation in real-time.

## Configuration
Modify backend configuration properties inside `backend/config.py` (e.g., threshold adjustments, video source, or toggling YOLO vs CSRNet architectures).
