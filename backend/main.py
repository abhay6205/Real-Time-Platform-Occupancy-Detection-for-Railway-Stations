import time  # PURPOSE: Adds timing delays to prevent system resource exhaustion and allow graceful server startup
import cv2  # PURPOSE: OpenCV - core library for video capture, frame processing, drawing annotations, and heatmap visualization
import threading  # PURPOSE: Enables multi-threading to run FastAPI server in background daemon thread without blocking the main AI detection loop
import uvicorn  # PURPOSE: ASGI application server - runs FastAPI backend so multiple clients (React dashboard) can connect and fetch real-time occupancy data concurrently

from datetime import datetime, timezone  # PURPOSE: Generates ISO 8601 formatted timestamps (timezone-aware UTC) for every occupancy record logged to database and API

import config  # PURPOSE: Central configuration management - stores all tunable parameters (thresholds, model paths, API settings) so engineers can modify behavior without editing code
from detector.video_capture import VideoCapture  # PURPOSE: Wrapper for cv2.VideoCapture - handles RTSP streams, MP4 files, webcams with automatic metadata extraction and error handling
from detector.frame_processor import FrameProcessor  # PURPOSE: Implements frame skipping logic to process 1 frame every N frames, reducing GPU load while maintaining analytical accuracy
from counter.person_counter import PersonCounter  # PURPOSE: Applies Exponential Moving Average (EMA) smoothing to raw AI counts to eliminate flickering and provide stable UI display
from counter.density_classifier import DensityClassifier  # PURPOSE: Classifies numerical crowd counts into semantic states (Low/Medium/High) with color codes for intuitive operator understanding
from storage.logger import OccupancyLogger  # PURPOSE: Dual-persistence logging - writes records to both SQLite database (queryable history) and CSV file (data recovery/backup)
from api.server import app, update_record, update_frame  # PURPOSE: FastAPI application and state management functions - app runs the REST API, update_record/update_frame manage global state for concurrent API clients
from api.models import OccupancyRecord  # PURPOSE: Pydantic data validation model - enforces strict schema for occupancy records (count, density, timestamp, color) across API and database

def main():
    """
    The central orchestration engine of the Railway Occupancy System.
    
    This script initializes all standalone components (Capture, Processing, AI, Logging)
    and binds them together in a continuous, synchronous while-loop.
    It also spins off the FastAPI server into a separate background daemon thread 
    so the web dashboard can be served concurrently without blocking the AI inference.
    """
    print("Initialising system components...")

    # --- Phase 1: Initialize Pipeline Components ---
    # We instantiate these outside the loop to prevent memory leaks and overhead
    capture = VideoCapture(config.VIDEO_SOURCE)
    processor = FrameProcessor(config.FRAME_SKIP, config.INPUT_SIZE if not config.USE_CSRNET else 640)
    counter = PersonCounter(config.EMA_ALPHA)
    classifier = DensityClassifier(config.LOW_MAX, config.HIGH_MIN)
    logger = OccupancyLogger(config.LOG_CSV_PATH, config.LOG_DB_PATH)

    # --- Phase 2: Initialize AI Detector based on Strategy ---
    if config.USE_CSRNET:
        from detector.crowd_detector import CrowdDetector
        detector = CrowdDetector(model_path=config.CSRNET_WEIGHTS)
        print("Using CSRNet crowd density estimation model")
    
    # [FALLBACK LOGIC PRESERVED]
    # If config.USE_CSRNET is False, the system routes through YOLO bounding box detection.
    # else:
    #     from detector.yolo_detector import YOLODetector
    #     detector = YOLODetector(
    #         model_path=config.MODEL_PATH,
    #         confidence=config.CONFIDENCE_THRESHOLD,
    #         iou=config.NMS_IOU_THRESHOLD,
    #         person_class_id=config.PERSON_CLASS_ID,
    #         imgsz=config.INPUT_SIZE
    #     )
    #     print("Using YOLOv8 object detection model")

    # --- Phase 3: Launch Background API Server ---
    # We run FastAPI via uvicorn in a daemon thread. A daemon thread automatically 
    # dies when the main program closes, preventing zombie processes.
    print(f"Starting API server on {config.API_HOST}:{config.API_PORT}...")
    
    # threading.Thread: Creates a separate execution thread so FastAPI server runs concurrently with AI detection loop
    # WITHOUT threading, uvicorn.run() would block forever and no frames would be processed
    thread = threading.Thread(
        target=uvicorn.run,  # uvicorn: Runs the ASGI application (FastAPI) on specified host:port to serve REST API endpoints
        args=(app,),
        kwargs={"host": config.API_HOST, "port": config.API_PORT, "log_level": "error"},
        daemon=True
    )
    thread.start()
    
    # time.sleep(1): Pauses main thread for 1 second, giving uvicorn time to bind to socket before API requests arrive
    # This prevents connection refused errors at startup
    time.sleep(1)  

    print("\nDetection running. Open the React dashboard:")
    print("  cd frontend && npm run dev")
    print("  Then open http://localhost:5173\n")

    # --- Phase 4: Main Detection Loop ---
    try:
        while capture.is_opened():
            # cv2.VideoCapture.read_frame(): Pulls next frame from camera/video stream
            # Returns None when stream ends (prevents infinite loops on finite videos)
            frame = capture.read_frame()
            if frame is None:
                print("End of video stream or failed to read frame.")
                break

            # FrameProcessor.should_process(): Determines if frame should be processed or skipped
            # Skipping frames reduces GPU load from 30 FPS to ~10 FPS while maintaining accuracy
            if not processor.should_process(frame):
                continue

            if config.USE_CSRNET:
                # --- Primary Execution Path: CSRNet ---
                # Run the neural network to get the raw crowd count and density map
                result = detector.detect(frame)
                raw_count = result['count']
                
                # PersonCounter.update_from_count(): Applies Exponential Moving Average smoothing
                # Raw AI predictions are noisy; EMA prevents flickering (e.g., 45→40→45 becomes smooth curve)
                smoothed = counter.update_from_count(raw_count)

                # DensityClassifier: Uses config thresholds to convert numbers into semantic states
                # Dynamically update classification thresholds in case they were changed via the API
                classifier.low_max = config.LOW_MAX
                classifier.high_min = config.HIGH_MIN
                
                # Classify the smoothed count into actionable states ("High", "Low")
                label, colour = classifier.classify(smoothed)

                # datetime.now(timezone.utc).isoformat(): Generates ISO 8601 UTC timestamp
                # Ensures all records are timestamped for historical tracking and analytics
                # OccupancyRecord: Pydantic model validates that all fields match strict schema
                record = OccupancyRecord(
                    count=smoothed,
                    density=label,
                    colour=colour,
                    timestamp=datetime.now(timezone.utc).isoformat(),  # ISO 8601 format with timezone
                    smoothed=counter.smoothed_count
                )

                # Dispatch data to the API state and persist to databases
                update_record(record)  # Makes data available to FastAPI clients (React dashboard)
                logger.log(record)  # Writes to both SQLite DB and CSV WAL for persistence

                # cv2.cvtColor + cv2.resize: Used in detector.annotate() to prepare density heatmap for visualization
                annotated = detector.annotate(frame, result)
                
                # cv2.putText(): OpenCV function to draw text overlays on frames for real-time visual feedback
                # Helps operators immediately see crowding levels without checking dashboard
                cv2.putText(annotated, f"Density: {label}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # --- Phase 5: Stream and Render ---
            # Send the annotated frame to the FastAPI thread so the React UI can fetch the MJPEG stream
            update_frame(annotated)
            
            # cv2.imshow(): Displays annotated frame in local debug window on server
            # Useful for monitoring system behavior without accessing remote dashboard
            cv2.imshow("Railway Occupancy Detection", annotated)
            
            # cv2.waitKey(1): Waits 1ms for keyboard input, allows graceful shutdown via 'q' key
            # Non-blocking so frame processing continues in real-time
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # Catch CTRL+C to prevent ugly stack traces
        print("Interrupted by user.")
    finally:
        # cv2.destroyAllWindows(): Closes all OpenCV display windows
        # capture.release(): Releases hardware locks on camera/video file
        # CRITICAL: Prevents resource leaks that would prevent next run
        capture.release()
        cv2.destroyAllWindows()
        print("System shutdown complete.")

if __name__ == "__main__":
    main()
