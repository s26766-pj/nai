"""
Drawing Module
Handles drawing of crosshair and messages on frames
"""

import cv2


def draw_crosshair(frame, face_rect):
    """
    Draw turquoise crosshair between eyes on detected face
    
    Args:
        frame: OpenCV image frame (numpy array)
        face_rect: Tuple (x, y, w, h, confidence) representing face bounding box
    
    Returns:
        Tuple (center_x, eye_level_y): Position of the crosshair center
    """
    x, y, w, h, confidence = face_rect
    
    # Calculate position between eyes
    # Eyes are typically at about 1/3 from the top of the face bounding box
    # Center horizontally, but position vertically at eye level
    center_x = x + w // 2
    eye_level_y = y + int(h * 0.35)  # Approximately at eye level (1/3 from top)
    
    # Crosshair size (proportional to face size)
    # Make it larger and more prominent
    size = max(w, h) // 4
    
    # Turquoise color (BGR format: bright cyan/turquoise)
    # Turquoise is approximately (208, 224, 64) in BGR, but we'll use a brighter version
    crosshair_color = (255, 200, 0)  # Bright turquoise/cyan in BGR
    shadow_color = (50, 50, 50)  # Dark gray for shadow
    
    # Thickness of crosshair bars (thicker for game-like appearance)
    thickness = max(3, int(min(w, h) / 80))  # Proportional thickness, minimum 3
    
    # Draw shadow first (slightly offset for 3D effect)
    shadow_offset = 2
    
    # Horizontal line shadow
    cv2.line(frame, 
            (center_x - size + shadow_offset, eye_level_y + shadow_offset), 
            (center_x + size + shadow_offset, eye_level_y + shadow_offset), 
            shadow_color, thickness)
    
    # Vertical line shadow
    cv2.line(frame, 
            (center_x + shadow_offset, eye_level_y - size + shadow_offset), 
            (center_x + shadow_offset, eye_level_y + size + shadow_offset), 
            shadow_color, thickness)
    
    # Draw main crosshair bars (turquoise)
    # Horizontal line
    cv2.line(frame, 
            (center_x - size, eye_level_y), 
            (center_x + size, eye_level_y), 
            crosshair_color, thickness)
    
    # Vertical line
    cv2.line(frame, 
            (center_x, eye_level_y - size), 
            (center_x, eye_level_y + size), 
            crosshair_color, thickness)
    
    return (center_x, eye_level_y)


def draw_message(frame, message):
    """
    Draw message on the frame
    
    Args:
        frame: OpenCV image frame
        message: Message to display
    """
    h, w = frame.shape[:2]
    
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 4
    (text_width, text_height), baseline = cv2.getTextSize(message, font, font_scale, thickness)
    
    # Position text in center-top of frame
    text_x = (w - text_width) // 2
    text_y = 80
    
    # Draw black background rectangle for better visibility
    padding = 20
    cv2.rectangle(frame,
                 (text_x - padding, text_y - text_height - padding),
                 (text_x + text_width + padding, text_y + baseline + padding),
                 (0, 0, 0), -1)
    
    # Draw text (red for "BACK AWAY!", yellow for "HANDS UP!")
    color = (0, 0, 255) if "BACK" in message else (0, 255, 255)  # Red or Yellow
    cv2.putText(frame, message,
               (text_x, text_y),
               font, font_scale,
               color,
               thickness,
               cv2.LINE_AA)


def draw_counter(frame, count):
    """
    Draw counter in the bottom-left corner of the frame
    
    Args:
        frame: OpenCV image frame
        count: Counter value to display
    """
    h, w = frame.shape[:2]
    
    # Counter text
    counter_text = f"HANDS UP: {count}"
    
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(counter_text, font, font_scale, thickness)
    
    # Position text in bottom-left corner
    text_x = 10
    text_y = h - 10
    
    # Draw semi-transparent black background rectangle
    padding = 5
    cv2.rectangle(frame,
                 (text_x - padding, text_y - text_height - padding),
                 (text_x + text_width + padding, text_y + baseline + padding),
                 (0, 0, 0), -1)
    
    # Draw white text
    cv2.putText(frame, counter_text,
               (text_x, text_y),
               font, font_scale,
               (255, 255, 255),  # White color
               thickness,
               cv2.LINE_AA)


def draw_target_eliminated(frame):
    """
    Draw "TARGET ELIMINATED" message permanently in the center of the frame
    
    Args:
        frame: OpenCV image frame
    """
    h, w = frame.shape[:2]
    
    # Text to display (split into two lines)
    line1 = "TARGET"
    line2 = "ELIMINATED"
    
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0
    thickness = 6
    
    # Get text sizes for both lines
    (text_width1, text_height1), baseline1 = cv2.getTextSize(line1, font, font_scale, thickness)
    (text_width2, text_height2), baseline2 = cv2.getTextSize(line2, font, font_scale, thickness)
    
    # Use the wider line for centering
    max_width = max(text_width1, text_width2)
    line_spacing = 20
    
    # Position text in center of frame
    text_x1 = (w - text_width1) // 2
    text_x2 = (w - text_width2) // 2
    text_y1 = (h - text_height1 - text_height2 - line_spacing) // 2 + text_height1
    text_y2 = text_y1 + text_height2 + line_spacing
    
    # Draw black background rectangle for better visibility
    padding = 30
    total_height = text_height1 + text_height2 + line_spacing + baseline1 + baseline2
    cv2.rectangle(frame,
                 ((w - max_width) // 2 - padding, text_y1 - text_height1 - padding),
                 ((w + max_width) // 2 + padding, text_y2 + baseline2 + padding),
                 (0, 0, 0), -1)
    
    # Draw red text
    color = (0, 0, 255)  # Red color in BGR
    cv2.putText(frame, line1,
               (text_x1, text_y1),
               font, font_scale,
               color,
               thickness,
               cv2.LINE_AA)
    
    cv2.putText(frame, line2,
               (text_x2, text_y2),
               font, font_scale,
               color,
               thickness,
               cv2.LINE_AA)

