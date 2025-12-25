"""
Posture Detection Module using MediaPipe Pose
Uses MediaPipe 0.10+ Tasks API for pose detection
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request


class PostureDetector:
    """
    Posture detection using MediaPipe Pose Landmarker
    """
    
    def __init__(self, 
                 model_path="models/pose_landmarker_lite.task",
                 num_poses=1,
                 min_pose_detection_confidence=0.5,
                 min_pose_presence_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize posture detector
        
        Args:
            model_path: Path to pose landmarker model file
            num_poses: Maximum number of poses to detect
            min_pose_detection_confidence: Minimum confidence for pose detection
            min_pose_presence_confidence: Minimum confidence for pose presence
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.model_path = model_path
        self.num_poses = num_poses
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Download model if not present
        self._download_model()
        
        # Initialize landmarker
        self._init_landmarker()
        
        # Pose connections for drawing skeleton
        self.POSE_CONNECTIONS = [
            (11, 13), (13, 15),  # left arm
            (12, 14), (14, 16),  # right arm
            (11, 12),            # shoulders
            (11, 23), (12, 24),  # torso
            (23, 24),            # hips
            (23, 25), (25, 27),  # left leg
            (24, 26), (26, 28)   # right leg
        ]
        
        self.timestamp = 0
        print("Posture detector initialized successfully")
    
    def _download_model(self):
        """Download pose landmarker model if not present"""
        model_dir = os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else "models"
        os.makedirs(model_dir, exist_ok=True)
        
        if not os.path.exists(self.model_path):
            print("Downloading pose landmarker model...")
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_lite/float16/latest/"
                "pose_landmarker_lite.task"
            )
            try:
                urllib.request.urlretrieve(model_url, self.model_path)
                print(f"Model downloaded to {self.model_path}")
            except Exception as e:
                print(f"Failed to download model: {e}")
                raise
    
    def _init_landmarker(self):
        """Initialize MediaPipe Pose Landmarker"""
        BaseOptions = python.BaseOptions
        PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        VisionRunningMode = vision.RunningMode
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        self.landmarker = PoseLandmarker.create_from_options(options)
    
    def detect_pose(self, frame):
        """
        Detect pose in the frame
        
        Args:
            frame: BGR image frame (numpy array)
        
        Returns:
            List of pose landmarks. Each pose is a list of (x, y) coordinates
            Returns empty list if no pose detected
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        self.timestamp += 1
        result = self.landmarker.detect_for_video(mp_image, self.timestamp)
        
        poses = []
        if result.pose_landmarks:
            for pose_landmarks in result.pose_landmarks:
                points = []
                for landmark in pose_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))
                poses.append(points)
        
        return poses
    
    def draw_pose(self, frame, poses):
        """
        Draw pose landmarks and skeleton on the frame
        
        Args:
            frame: BGR image frame (numpy array)
            poses: List of pose landmarks (list of (x, y) tuples)
        
        Returns:
            Frame with pose drawn (modifies frame in place)
        """
        for points in poses:
            # Draw landmarks (joints)
            for point in points:
                if point:
                    cv2.circle(frame, point, 4, (0, 255, 0), -1)
            
            # Draw skeleton connections
            for a, b in self.POSE_CONNECTIONS:
                if a < len(points) and b < len(points) and points[a] and points[b]:
                    cv2.line(frame, points[a], points[b], (255, 0, 255), 2)
        
        return frame
    
    def get_hand_positions(self, poses):
        """
        Get hand positions from pose landmarks
        
        Args:
            poses: List of pose landmarks
        
        Returns:
            List of hand positions. Each hand is (x, y) of wrist position
            Index 15 = left wrist, Index 16 = right wrist
        """
        hands = []
        for pose in poses:
            if len(pose) > 16:
                # Left wrist (index 15)
                if pose[15]:
                    hands.append(pose[15])
                # Right wrist (index 16)
                if pose[16]:
                    hands.append(pose[16])
        return hands
    
    def cleanup(self):
        """Release resources"""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()

