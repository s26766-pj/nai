"""
Sound Module
Handles playing sound effects
"""

import os


def play_sound(file_path):
    """
    Play a sound file
    
    Args:
        file_path: Path to the sound file (mp3, wav, etc.)
    
    Returns:
        bool: True if sound was played successfully, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"Warning: Sound file not found: {file_path}")
        return False
    
    try:
        # Try using pygame for cross-platform sound playback
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            return True
        except ImportError:
            # Fallback to platform-specific methods
            import platform
            system = platform.system()
            
            if system == 'Windows':
                # Windows: use winsound for wav files, or subprocess for mp3
                import subprocess
                try:
                    # Try using Windows Media Player
                    subprocess.Popen(['start', 'wmplayer', file_path], shell=True)
                    return True
                except:
                    # Try using default player
                    os.startfile(file_path)
                    return True
            elif system == 'Darwin':  # macOS
                import subprocess
                subprocess.Popen(['afplay', file_path])
                return True
            else:  # Linux
                import subprocess
                subprocess.Popen(['aplay', file_path])
                return True
    except Exception as e:
        print(f"Error playing sound: {e}")
        return False


def play_target_eliminated_sound():
    """
    Play the gun shot sound when target is eliminated
    
    Returns:
        bool: True if sound was played successfully
    """
    sound_file = "media/gun-shots-230534.mp3"
    return play_sound(sound_file)

