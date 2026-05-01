import pytest
import numpy as np
from detector.yolo_detector import YOLODetector

def test_yolo_detector():
    detector = YOLODetector(model_path="yolov8n.pt", confidence=0.40, iou=0.45, person_class_id=0)

    # Create a blank 640x640 black numpy image (BGR format)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    # Call detect
    detections = detector.detect(frame)

    # Assert return value is a list (even if empty)
    assert isinstance(detections, list)

    # Assert annotate returns a numpy array of same shape
    annotated = detector.annotate(frame, detections)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (640, 640, 3)
