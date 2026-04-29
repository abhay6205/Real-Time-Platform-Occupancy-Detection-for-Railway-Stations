import cv2
import numpy as np

class FrameProcessor:
    """
    Handles frame skipping and resolution downscaling.
    
    Why skip frames?
    A standard CCTV camera records at 30 to 60 FPS. Running a heavy Neural Network 
    like CSRNet on every single frame will overwhelm even high-end GPUs, causing massive lag.
    Because crowd density doesn't change drastically in 1/30th of a second, we can 
    safely process only 1 out of every N frames (e.g., process 1, skip 3) to maintain
    real-time system performance without losing critical analytical data.
    """

    def __init__(self, skip_interval: int, input_size: int):
        """
        Initialises the processor logic.
        
        :param skip_interval: Number of frames to wait before running full AI inference.
                              e.g., skip_interval=4 means process 1 frame, skip the next 3.
        :param input_size: The target dimension for AI processing (primarily used for YOLO, 
                           as CSRNet determines scale dynamically in its own class).
        """
        self.skip_interval = skip_interval
        self.input_size = input_size
        self.counter = 0

    def should_process(self, frame: np.ndarray) -> bool:
        """
        Determines whether the current frame should be fed into the AI models or discarded.
        
        :param frame: The raw frame from the video source.
        :return: True if the frame should be processed, False to skip.
        """
        self.counter += 1
        # Modulo arithmetic ensures we only hit True every Nth frame.
        return self.counter % self.skip_interval == 0

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resizes frame to a fixed square dimension.
        (Note: Currently used mainly if the YOLO fallback is enabled, as YOLO prefers fixed 
        resolutions like 640x640 or 1280x1280. CSRNet scales dynamically based on aspect ratio).
        """
        return cv2.resize(frame, (self.input_size, self.input_size))
