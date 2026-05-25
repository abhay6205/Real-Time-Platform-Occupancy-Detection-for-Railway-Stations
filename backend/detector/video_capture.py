import cv2  # PURPOSE: OpenCV - handles video capture from cameras, RTSP streams, and video files; extracts metadata
import logging  # PURPOSE: Logging framework - records video connection status and errors for debugging

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
        # cv2.VideoCapture(source): Opens connection to camera, video file, or RTSP stream
        # 0 = system default webcam, positive int = specific camera ID, string = filepath or RTSP URL
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            # cv2.VideoCapture.isOpened(): Checks if video source was successfully opened
            # Fails silently if file doesn't exist or RTSP stream is offline
            raise RuntimeError(f"Could not open video source: {source}. Ensure the file exists or the stream is online.")

        # Extract underlying stream metadata using cv2 properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Resolution width
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Resolution height
        
        # logging: Records connection details to console for debugging and system monitoring
        logging.info(f"Successfully opened video source '{source}' | Resolution: {width}x{height} | FPS: {fps}")

    def read_frame(self):
        """
        Pulls the next frame from the hardware buffer or video file.
        
        :return: A numpy BGR array representing the image, or None if the stream has ended.
        """
        # cv2.VideoCapture.read(): Fetches next frame from video buffer
        # Returns (success_flag, frame_numpy_array)
        ret, frame = self.cap.read()
        if not ret:
            # Returning None explicitly signals the main loop to break/stop processing
            # ret=False means end of stream, corrupted frame, or device disconnected
            return None
        return frame

    def release(self):
        """
        Safely frees the hardware resources or file locks. 
        Critical to prevent memory leaks or locked hardware when shutting down the system.
        """
        # cv2.VideoCapture.release(): Closes the video capture device
        # ESSENTIAL: Prevents hardware locks (camera stuck in use) on next run
        # Also frees memory allocated for buffering
        self.cap.release()

    def is_opened(self) -> bool:
        """
        Checks if the stream is currently alive.
        """
        # cv2.VideoCapture.isOpened(): Returns True if stream is connected, False if closed/failed
        return self.cap.isOpened()
