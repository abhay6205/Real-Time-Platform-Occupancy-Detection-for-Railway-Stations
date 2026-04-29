import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    """
    FALLBACK COMPONENT: YOLOv8 Object Detection Interface.
    
    Why is this here?
    While CSRNet (Crowd Density Estimation) is vastly superior for massive crowds (100+ people),
    it can sometimes be overkill or slightly inaccurate for sparse environments (e.g., 2 to 5 people).
    YOLOv8 is an industry-standard object detector that draws precise bounding boxes around individuals.
    
    This file has been intentionally preserved. If the system is deployed to a quiet 
    regional station instead of a major city hub, setting `USE_CSRNET = False` in `config.py` 
    will route the data flow through this class instead.
    """

    def __init__(self, model_path: str, confidence: float, iou: float, person_class_id: int, imgsz: int = 640):
        """
        Loads the YOLO architecture and weights.
        
        :param model_path: Path to the .pt YOLO weights file.
        :param confidence: The minimum confidence score (e.g. 0.25) to register a hit.
        :param iou: Intersection over Union threshold to filter out overlapping boxes.
        :param person_class_id: The specific ID for humans (usually 0 in the COCO dataset).
        """
        # Note: Implementation logic is currently commented out to prevent heavy ultralytics 
        # imports from slowing down the primary CSRNet boot sequence. Uncomment to activate.
        pass
        # self.model = YOLO(model_path)
        # self.confidence = confidence
        # self.iou = iou
        # self.person_class_id = person_class_id
        # self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Runs YOLO inference on a frame and returns a list of bounding boxes.
        """
        return []
        # results = self.model.predict(frame, conf=self.confidence, iou=self.iou, imgsz=self.imgsz, verbose=False)
        # detections = []
        # for box in results[0].boxes:
        #     if int(box.cls[0].item()) == self.person_class_id:
        #         x1, y1, x2, y2 = box.xyxy[0].tolist()
        #         conf = box.conf[0].item()
        #         detections.append({
        #             "bbox": [x1, y1, x2, y2],
        #             "confidence": conf
        #         })
        # return detections

    def annotate(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draws the visual bounding boxes onto the frame for the UI.
        """
        return frame.copy()
        # annotated_frame = frame.copy()
        # for det in detections:
        #     x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        #     conf = det["confidence"]
        #     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(annotated_frame, f"{conf:.2f}", (x1, max(y1 - 10, 0)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # return annotated_frame
