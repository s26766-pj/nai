"""
Authors:
Kamil Suchomski
Kamil Koniak

Camera Stream Application with Posture Detection
For "Baba Jaga patrzy" game prototype
"""

import cv2
import sys
import time
from camera_support import CameraDevice, list_available_cameras
from posture_detection import PostureDetector
from draw import draw_crosshair, draw_message, draw_target_eliminated
from sound import play_target_eliminated_sound


class CameraStream:
    def __init__(self, camera_index=0):
        """
        Initialize camera stream with posture detection
        
        Args:
            camera_index: Index of the camera (usually 0 for default webcam)
        """
        self.camera_index = camera_index
        self.camera_device = CameraDevice(camera_index)
        
        # Initialize posture detector
        self.posture_detector = PostureDetector(
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize face detector (Haar cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")
        print("Face detector initialized successfully")
    
    def _check_hands_raised(self, poses, eye_level_positions, frame_width):
        """
        Check if hands (wrists) are raised above eye level
        
        Args:
            poses: List of pose landmarks (list of (x, y) tuples)
            eye_level_positions: List of eye level y-coordinates
            frame_width: Width of the frame (for adjusting coordinates after flip)
        
        Returns:
            bool: True if any hand is raised above eye level
        """
        if not poses or not eye_level_positions:
            return False
        
        # MediaPipe pose landmark indices:
        # 15 = Left wrist
        # 16 = Right wrist
        # Note: After flipping, left becomes right and vice versa
        
        # Get the lowest eye level (highest on screen, smallest y value)
        min_eye_level = min(eye_level_positions)
        
        # Check all poses
        for pose in poses:
            if len(pose) > 16:
                # Get wrist positions (after flip, indices are swapped)
                # Original left wrist (15) is now on the right side after flip
                # Original right wrist (16) is now on the left side after flip
                try:
                    # Safely get wrist positions
                    left_wrist_original = pose[15] if 15 < len(pose) and pose[15] is not None else None
                    right_wrist_original = pose[16] if 16 < len(pose) and pose[16] is not None else None
                    
                    # Adjust x coordinates for flipped frame
                    if left_wrist_original is not None and isinstance(left_wrist_original, (tuple, list)) and len(left_wrist_original) >= 2:
                        flipped_x = frame_width - int(left_wrist_original[0])
                        left_wrist_y = int(left_wrist_original[1])
                        # Check if wrists are above eye level (lower y value = higher on screen)
                        if left_wrist_y < min_eye_level:
                            return True
                    
                    if right_wrist_original is not None and isinstance(right_wrist_original, (tuple, list)) and len(right_wrist_original) >= 2:
                        flipped_x = frame_width - int(right_wrist_original[0])
                        right_wrist_y = int(right_wrist_original[1])
                        # Check if wrists are above eye level (lower y value = higher on screen)
                        if right_wrist_y < min_eye_level:
                            return True
                except (IndexError, TypeError, ValueError) as e:
                    # Skip this pose if there's an error accessing landmarks
                    continue
        
        return False
    
    def start_stream(self, width=640, height=480, fps=30):
        """
        Start camera stream with posture detection
        
        Args:
            width: Frame width (default: 640)
            height: Frame height (default: 480)
            fps: Frames per second (default: 30)
        """
        # Open camera device
        success, backend_used, properties = self.camera_device.open(width, height, fps)
        
        if not success:
            raise RuntimeError("Failed to open camera device")
        
        print(f"Camera stream started (Backend: {backend_used})")
        print(f"Camera {self.camera_index}: {properties['width']}x{properties['height']} @ {properties['fps']} FPS")
        print("Posture detection enabled")
        print("Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        message_start_time = None
        current_message = None
        message_duration = 3.0  # Message duration in seconds
        hands_up_counter = 0  # Counter for "HANDS UP!" displays
        sound_played = False  # Track if sound has been played
        
        while True:
            ret, frame = self.camera_device.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect poses
            poses = self.posture_detector.detect_pose(frame)
            
            # Draw poses on frame
            self.posture_detector.draw_pose(frame, poses)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Flip frame horizontally for mirror effect (before drawing crosshair)
            frame = cv2.flip(frame, 1)
            
            # Adjust face coordinates for flipped frame and calculate eye level positions
            # Also track eye level positions for hand detection
            h, w = frame.shape[:2]
            eye_level_positions = []
            face_rects = []
            
            for (x, y, face_w, face_h) in faces:
                # Adjust x coordinate for flipped frame
                flipped_x = w - x - face_w
                # Convert to format expected by draw_crosshair: (x, y, w, h, confidence)
                face_rect = (flipped_x, y, face_w, face_h, 1.0)
                face_rects.append(face_rect)
                
                # Calculate eye level y position (same as crosshair calculation)
                eye_level_y = y + int(face_h * 0.35)  # Approximately at eye level
                eye_level_positions.append(eye_level_y)
            
            # Check if target is eliminated (counter >= 20)
            if hands_up_counter >= 30:
                # Show only "TARGET ELIMINATED" permanently
                draw_target_eliminated(frame)
                
                # Play sound once when target is eliminated
                if not sound_played:
                    play_target_eliminated_sound()
                    sound_played = True
            else:
                # Normal operation - check hands and show messages
                # Check if hands are raised above eye level
                hands_raised = self._check_hands_raised(poses, eye_level_positions, w)
                
                # Draw crosshair only if hands are NOT raised
                if not hands_raised:
                    for face_rect in face_rects:
                        draw_crosshair(frame, face_rect)
                
                # Display messages based on hand position with timeout
                # Use len() to check NumPy arrays instead of direct boolean check
                current_time = time.time()
                
                if len(faces) > 0 and len(eye_level_positions) > 0:
                    # Determine what message should be shown
                    new_message = "BACK AWAY!" if hands_raised else "HANDS UP!"
                    
                    # If message changed or no message is currently shown, start timer
                    if new_message != current_message:
                        # Increment counter when "HANDS UP!" message appears
                        if new_message == "HANDS UP!":
                            hands_up_counter += 1
                        current_message = new_message
                        message_start_time = current_time
                    
                    # Show message only if less than timeout duration has passed
                    if message_start_time is not None:
                        elapsed_time = current_time - message_start_time
                        if elapsed_time < message_duration:
                            draw_message(frame, current_message)
                        else:
                            # Message timeout - clear it
                            current_message = None
                            message_start_time = None
                else:
                    # No face detected - clear message
                    current_message = None
                    message_start_time = None
            
            # Display frame
            cv2.imshow('Baba Jaga patrzy', frame)
            
            # Handle keyboard input
            # Use waitKey with a small delay to ensure key presses are captured
            key = cv2.waitKey(10) & 0xFF
            # Check for 'q', 'Q', or ESC (27) to quit
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 is ESC key
                break
            elif key == ord('s') or key == ord('S'):
                filename = f'frame_{frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
                frame_count += 1
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        self.posture_detector.cleanup()
        self.camera_device.release()
        cv2.destroyAllWindows()
        print("Camera stream closed")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Camera stream with posture detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available cameras
  python camera_stream.py --list-cameras
  
  # Use default camera (index 0)
  python camera_stream.py
  
  # Use USB camera at index 1
  python camera_stream.py --camera 1
  
  # Use USB camera with custom resolution
  python camera_stream.py --camera 1 --width 1280 --height 720
        """
    )
    parser.add_argument('--camera', type=int, default=0,
                       help='USB camera index (default: 0). Use --list-cameras to find available cameras')
    parser.add_argument('--width', type=int, default=640,
                       help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Frame height (default: 480)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--list-cameras', action='store_true',
                       help='List all available cameras and exit')
    
    args = parser.parse_args()
    
    if args.list_cameras:
        cameras = list_available_cameras()
        if cameras:
            print(f"\nFound {len(cameras)} available camera(s): {cameras}")
            print(f"\nUse: python camera_stream.py --camera {cameras[0]}")
        else:
            print("\nNo cameras found. Make sure your USB camera is connected.")
        sys.exit(0)
    
    try:
        stream = CameraStream(camera_index=args.camera)
        stream.start_stream(width=args.width, height=args.height, fps=args.fps)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

