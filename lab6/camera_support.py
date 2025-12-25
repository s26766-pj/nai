"""
Camera Device Support Module
Handles USB camera device operations, backend selection, and camera enumeration
"""

import cv2
import platform


def list_available_cameras(max_test=5):
    """
    List all available cameras by testing indices
    
    Args:
        max_test: Maximum camera index to test (default: 5)
    
    Returns:
        List of available camera indices
    """
    available = []
    print("Scanning for available cameras...")
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to confirm it's working
            ret, _ = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  Camera {i}: {width}x{height} - Available")
                available.append(i)
            cap.release()
        else:
            cap.release()
    
    return available


class CameraDevice:
    """
    Handles camera device operations including opening, configuration, and frame capture
    """
    
    def __init__(self, camera_index=0):
        """
        Initialize camera device
        
        Args:
            camera_index: Index of the camera (usually 0 for default webcam)
        """
        self.camera_index = camera_index
        self.cap = None
        self.backend_used = None
    
    def open(self, width=640, height=480, fps=30):
        """
        Open camera with appropriate backend for USB cameras
        
        Args:
            width: Frame width (default: 640)
            height: Frame height (default: 480)
            fps: Frames per second (default: 30)
        
        Returns:
            Tuple (success: bool, backend_name: str, actual_properties: dict)
        """
        # Try different backends for better USB camera support
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),  # Windows
            (cv2.CAP_V4L2, "V4L2"),        # Linux
            (cv2.CAP_ANY, "Default")        # Fallback
        ]
        
        self.cap = None
        self.backend_used = None
        
        # Try to open camera with different backends
        for backend_id, backend_name in backends:
            try:
                if platform.system() == 'Windows':
                    # On Windows, try DirectShow first for USB cameras
                    self.cap = cv2.VideoCapture(self.camera_index, backend_id)
                else:
                    # On Linux/Mac, try V4L2 or default
                    self.cap = cv2.VideoCapture(self.camera_index, backend_id)
                
                if self.cap.isOpened():
                    self.backend_used = backend_name
                    break
            except:
                continue
        
        if self.cap is None or not self.cap.isOpened():
            available = list_available_cameras()
            available_str = str(available) if available else "None found"
            raise RuntimeError(
                f"Failed to open USB camera {self.camera_index}.\n"
                f"Available cameras: {available_str}\n"
                f"Try: python camera_stream.py --list-cameras"
            )
        
        # Set camera properties for USB cameras
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Get actual camera properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        properties = {
            'width': actual_width,
            'height': actual_height,
            'fps': actual_fps
        }
        
        return True, self.backend_used, properties
    
    def read(self):
        """
        Read a frame from the camera
        
        Returns:
            Tuple (success: bool, frame: numpy.ndarray or None)
        """
        if self.cap is None:
            return False, None
        
        return self.cap.read()
    
    def is_opened(self):
        """
        Check if camera is opened
        
        Returns:
            bool: True if camera is opened, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.backend_used = None

