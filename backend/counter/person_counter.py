class PersonCounter:
    """
    Counts detections and applies Exponential Moving Average (EMA) smoothing.
    
    Why EMA? 
    Raw AI predictions (whether CSRNet density sums or YOLO bounding box counts) 
    are inherently noisy. A person might be occluded for a single frame, dropping 
    the count from 45 to 40, and back to 45 on the next frame. If we displayed 
    raw counts directly to the UI, the numbers would flicker rapidly, confusing the user.
    EMA solves this by heavily weighting the historical count and only slightly 
    nudging it towards the new raw count, resulting in a smooth, stable curve.
    """

    def __init__(self, ema_alpha: float):
        """
        Initialises the counter.
        :param ema_alpha: The smoothing factor between 0.0 and 1.0. 
                          A lower value (e.g., 0.1) means extreme smoothness but slow reaction time.
                          A higher value (e.g., 0.8) means fast reaction but higher jitter.
        """
        self.ema_alpha = ema_alpha
        self.smoothed_count = 0.0

    def update(self, detections: list) -> int:
        """
        Updates count using EMA smoothing from a list of bounding box detections.
        (Primarily used as a fallback if switching back to YOLO object detection).
        
        :param detections: A list of dictionary detections from the YOLO model.
        :return: The smoothed, rounded integer count.
        """
        raw_count = len(detections)
        self.smoothed_count = (self.ema_alpha * raw_count) + ((1 - self.ema_alpha) * self.smoothed_count)
        return int(round(self.smoothed_count))

    def update_from_count(self, raw_count: int) -> int:
        """
        Updates count using EMA smoothing from a raw integer count.
        (This is the primary method used by CSRNet, as it returns a numeric density sum, not bounding boxes).
        
        :param raw_count: The instantaneous crowd count estimated by CSRNet.
        :return: The smoothed, rounded integer count.
        """
        self.smoothed_count = (self.ema_alpha * raw_count) + ((1 - self.ema_alpha) * self.smoothed_count)
        return int(round(self.smoothed_count))

    def reset(self):
        """
        Resets smoothed count to 0.0.
        Useful when switching video streams or restarting a session.
        """
        self.smoothed_count = 0.0
