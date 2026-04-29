import cv2
import logging

class VideoCapture:
    """
    A robust wrapper around OpenCV's VideoCapture functionality.
    
    Why use a wrapper?
    Raw OpenCV video initialization can fail silently or lack structured logging.
    This class encapsulates the connection logic, supports multiple source types 
    (webcams, RTSP streams, local MP4 files), and automatically extracts and logs
    critical metadata like Resolution and FPS upon connection.
    """

    def __init__(self, source: str | int):
        """
        Opens video source and logs connection details.
        
        :param source: Can be an integer (e.g., 0 for default webcam) or a string 
                       representing a file path ("videos/test.mp4") or an RTSP IP stream.
        """
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}. Ensure the file exists or the stream is online.")

        # Extract underlying stream metadata
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logging.info(f"Successfully opened video source '{source}' | Resolution: {width}x{height} | FPS: {fps}")

    def read_frame(self):
        """
        Pulls the next frame from the hardware buffer or video file.
        
        :return: A numpy BGR array representing the image, or None if the stream has ended.
        """
        ret, frame = self.cap.read()
        if not ret:
            # Returning None explicitly signals the main loop to break/stop processing
            return None
        return frame

    def release(self):
        """
        Safely frees the hardware resources or file locks. 
        Critical to prevent memory leaks or locked hardware when shutting down the system.
        """
        self.cap.release()

    def is_opened(self) -> bool:
        """
        Checks if the stream is currently alive.
        """
        return self.cap.isOpened()
