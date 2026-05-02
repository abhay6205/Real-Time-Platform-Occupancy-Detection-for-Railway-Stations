import time
import cv2
import threading
import uvicorn
from datetime import datetime, timezone

import config
from detector.video_capture import VideoCapture
from detector.frame_processor import FrameProcessor
from counter.person_counter import PersonCounter
from counter.density_classifier import DensityClassifier
from storage.logger import OccupancyLogger
from api.server import app, update_record, update_frame
from api.models import OccupancyRecord

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
    thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": config.API_HOST, "port": config.API_PORT, "log_level": "error"},
        daemon=True
    )
    thread.start()
    
    # Provide a brief grace period for the server socket to bind before hammering it with data
    time.sleep(1)  

    print("\nDetection running. Open the React dashboard:")
    print("  cd frontend && npm run dev")
    print("  Then open http://localhost:5173\n")

    # --- Phase 4: Main Detection Loop ---
    try:
        while capture.is_opened():
            # 1. Pull raw frame from camera/video
            frame = capture.read_frame()
            if frame is None:
                print("End of video stream or failed to read frame.")
                break

            # 2. Skip frames to maintain target FPS and prevent GPU overload
            if not processor.should_process(frame):
                continue

            if config.USE_CSRNET:
                # --- Primary Execution Path: CSRNet ---
                # Run the neural network to get the raw crowd count and density map
                result = detector.detect(frame)
                raw_count = result['count']
                
                # Apply Exponential Moving Average (EMA) to prevent UI flickering
                smoothed = counter.update_from_count(raw_count)

                # Dynamically update classification thresholds in case they were changed via the API
                classifier.low_max = config.LOW_MAX
                classifier.high_min = config.HIGH_MIN
                
                # Classify the smoothed count into actionable states ("High", "Low")
                label, colour = classifier.classify(smoothed)

                # Package the data into a strict schema
                record = OccupancyRecord(
                    count=smoothed,
                    density=label,
                    colour=colour,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    smoothed=counter.smoothed_count
                )

                # Dispatch data to the API state and persist to databases
                update_record(record)
                logger.log(record)

                # Visually blend the AI heatmap onto the OpenCV frame
                annotated = detector.annotate(frame, result)
                
                # Overlay real-time text warnings
                cv2.putText(annotated, f"Density: {label}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # [FALLBACK LOGIC PRESERVED]
            # --- Alternate Execution Path: YOLOv8 ---
            # else:
            #     detections = detector.detect(frame)
            #     count = counter.update(detections)
            # 
            #     classifier.low_max = config.LOW_MAX
            #     classifier.high_min = config.HIGH_MIN
            #     label, colour = classifier.classify(count)
            # 
            #     record = OccupancyRecord(
            #         count=count,
            #         density=label,
            #         colour=colour,
            #         timestamp=datetime.now(timezone.utc).isoformat(),
            #         smoothed=counter.smoothed_count
            #     )
            # 
            #     update_record(record)
            #     logger.log(record)
            # 
            #     annotated = detector.annotate(frame, detections)
            #     info_text = f"Count: {count} | Density: {label}"
            #     cv2.putText(annotated, info_text, (10, 40),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # --- Phase 5: Stream and Render ---
            # Send the annotated frame to the FastAPI thread so the React UI can fetch the MJPEG stream
            update_frame(annotated)
            
            # Show a local debug window
            cv2.imshow("Railway Occupancy Detection", annotated)
            
            # Allow graceful exit via the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # Catch CTRL+C to prevent ugly stack traces
        print("Interrupted by user.")
    finally:
        # Safely release hardware locks and close windows
        capture.release()
        cv2.destroyAllWindows()
        print("System shutdown complete.")

if __name__ == "__main__":
    main()
