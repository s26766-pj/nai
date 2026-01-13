"""
Autors:
Kamil Suchomski s21974
Kamil Koniak s26766

Snake Game for Reinforcement Learning
Based on original Snake Eater by Rajat Dipta Biswas
Adapted for RL training by removing fuzzy logic and adding RL interface

This module provides a clean Snake game implementation that can be used
with reinforcement learning agents.
"""

import pygame
import sys
import random
from enum import Enum


class Direction(Enum):
    UP = 'UP'
    DOWN = 'DOWN'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'


class SnakeGame:
    """Snake game environment for RL training."""
    
    def __init__(self, frame_size_x=720, frame_size_y=480, block_size=10, difficulty=120, display=True):
        """
        Initialize the Snake game.
        
        Args:
            frame_size_x: Width of the game window
            frame_size_y: Height of the game window
            block_size: Size of each block (snake segment and food)
            difficulty: FPS/difficulty level
            display: Whether to display the game window
        """
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.block_size = block_size
        self.difficulty = difficulty
        self.display = display
        
        # Initialize pygame if display is enabled
        if self.display:
            check_errors = pygame.init()
            if check_errors[1] > 0:
                print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
                sys.exit(-1)
            else:
                print('[+] Game successfully initialised')
            
            pygame.display.set_caption('Snake RL')
            self.game_window = pygame.display.set_mode((frame_size_x, frame_size_y))
            self.fps_controller = pygame.time.Clock()
        
        # Colors (only needed if display is enabled)
        if self.display:
            self.black = pygame.Color(0, 0, 0)
            self.white = pygame.Color(255, 255, 255)
            self.red = pygame.Color(255, 0, 0)
            self.green = pygame.Color(0, 255, 0)
        
        self.reset()
    
    def reset(self):
        """Reset the game to initial state."""
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]
        self.direction = Direction.RIGHT
        self.score = 0
        self.food_pos = self._generate_food_position()
        self.food_spawn = True
        self.game_over_flag = False
        return self._get_state()
    
    def _get_all_valid_positions(self):
        """Get all valid food positions."""
        # Cache is handled at instance level
        if not hasattr(self, '_cached_valid_positions'):
            valid_positions = []
            for x in range(2, (self.frame_size_x//self.block_size)-1):
                for y in range(2, (self.frame_size_y//self.block_size)-1):
                    valid_positions.append((x * self.block_size, y * self.block_size))
            self._cached_valid_positions = valid_positions
        return self._cached_valid_positions
    
    def _generate_food_position(self):
        """Generate a random valid food position."""
        valid_positions = self._get_all_valid_positions()
        snake_body_set = set(tuple(segment) for segment in self.snake_body)
        
        available_positions = [pos for pos in valid_positions if pos not in snake_body_set]
        
        if available_positions:
            return list(random.choice(available_positions))
        else:
            # Fallback - should rarely happen
            return list(random.choice(valid_positions))
    
    def _check_collision(self, point=None):
        """
        Check if there's a collision.
        
        Args:
            point: Point to check (defaults to snake head)
        
        Returns:
            bool: True if collision detected
        """
        if point is None:
            point = self.snake_pos
        
        # Check wall collision
        if (point[0] < 0 or point[0] >= self.frame_size_x or
            point[1] < 0 or point[1] >= self.frame_size_y):
            return True
        
        # Check body collision
        for block in self.snake_body[1:]:
            if point[0] == block[0] and point[1] == block[1]:
                return True
        
        return False
    
    def _get_state(self):
        """
        Get the current game state as a feature vector for RL.
        
        Returns:
            list: State vector with 11 features:
                [danger_straight, danger_right, danger_left,
                 direction_left, direction_right, direction_up, direction_down,
                 food_left, food_right, food_up, food_down]
        """
        # Get current head position
        head_x, head_y = self.snake_pos
        
        # Calculate danger directions (collision in next step)
        danger_straight = 0
        danger_right = 0
        danger_left = 0
        
        # Check straight ahead
        if self.direction == Direction.UP:
            next_pos = [head_x, head_y - self.block_size]
        elif self.direction == Direction.DOWN:
            next_pos = [head_x, head_y + self.block_size]
        elif self.direction == Direction.LEFT:
            next_pos = [head_x - self.block_size, head_y]
        else:  # RIGHT
            next_pos = [head_x + self.block_size, head_y]
        
        danger_straight = 1 if self._check_collision(next_pos) else 0
        
        # Check right turn
        if self.direction == Direction.UP:
            next_pos = [head_x + self.block_size, head_y]
        elif self.direction == Direction.DOWN:
            next_pos = [head_x - self.block_size, head_y]
        elif self.direction == Direction.LEFT:
            next_pos = [head_x, head_y - self.block_size]
        else:  # RIGHT
            next_pos = [head_x, head_y + self.block_size]
        
        danger_right = 1 if self._check_collision(next_pos) else 0
        
        # Check left turn
        if self.direction == Direction.UP:
            next_pos = [head_x - self.block_size, head_y]
        elif self.direction == Direction.DOWN:
            next_pos = [head_x + self.block_size, head_y]
        elif self.direction == Direction.LEFT:
            next_pos = [head_x, head_y + self.block_size]
        else:  # RIGHT
            next_pos = [head_x, head_y - self.block_size]
        
        danger_left = 1 if self._check_collision(next_pos) else 0
        
        # Current direction (one-hot encoding)
        direction_left = 1 if self.direction == Direction.LEFT else 0
        direction_right = 1 if self.direction == Direction.RIGHT else 0
        direction_up = 1 if self.direction == Direction.UP else 0
        direction_down = 1 if self.direction == Direction.DOWN else 0
        
        # Food direction
        food_x, food_y = self.food_pos
        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0
        
        return [
            danger_straight, danger_right, danger_left,
            direction_left, direction_right, direction_up, direction_down,
            food_left, food_right, food_up, food_down
        ]
    
    def step(self, action):
        """
        Execute one step in the game.
        
        Args:
            action: Action to take (0=straight, 1=right turn, 2=left turn)
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.game_over_flag:
            return self._get_state(), 0, True, {}
        
        # Update direction based on action
        # Action: 0 = straight, 1 = right turn, 2 = left turn
        if action == 1:  # Right turn
            if self.direction == Direction.UP:
                self.direction = Direction.RIGHT
            elif self.direction == Direction.RIGHT:
                self.direction = Direction.DOWN
            elif self.direction == Direction.DOWN:
                self.direction = Direction.LEFT
            else:  # LEFT
                self.direction = Direction.UP
        elif action == 2:  # Left turn
            if self.direction == Direction.UP:
                self.direction = Direction.LEFT
            elif self.direction == Direction.LEFT:
                self.direction = Direction.DOWN
            elif self.direction == Direction.DOWN:
                self.direction = Direction.RIGHT
            else:  # RIGHT
                self.direction = Direction.UP
        # action == 0 means continue straight (no direction change)
        
        # Move snake
        if self.direction == Direction.UP:
            self.snake_pos[1] -= self.block_size
        elif self.direction == Direction.DOWN:
            self.snake_pos[1] += self.block_size
        elif self.direction == Direction.LEFT:
            self.snake_pos[0] -= self.block_size
        else:  # RIGHT
            self.snake_pos[0] += self.block_size
        
        # Check for collision
        if self._check_collision():
            self.game_over_flag = True
            return self._get_state(), -10, True, {'score': self.score}
        
        # Check if food eaten
        reward = -0.1  # Small negative reward for each step (encourage efficiency)
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 1
            reward = 10  # Positive reward for eating food
            self.food_spawn = False
            self.snake_body.insert(0, list(self.snake_pos))
        else:
            self.snake_body.insert(0, list(self.snake_pos))
            self.snake_body.pop()
        
        # Spawn new food if needed
        if not self.food_spawn:
            self.food_pos = self._generate_food_position()
        self.food_spawn = True
        
        return self._get_state(), reward, self.game_over_flag, {'score': self.score}
    
    def render(self):
        """Render the game (only if display is enabled)."""
        if not self.display:
            return
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
        
        # Draw game
        self.game_window.fill(self.black)
        
        # Draw snake
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, self.green, 
                           pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
        
        # Draw food
        pygame.draw.rect(self.game_window, self.red, 
                        pygame.Rect(self.food_pos[0], self.food_pos[1], 
                                  self.block_size, self.block_size))
        
        # Draw score
        score_font = pygame.font.SysFont('consolas', 20)
        score_surface = score_font.render('Score : ' + str(self.score), True, self.white)
        score_rect = score_surface.get_rect()
        score_rect.midtop = (self.frame_size_x/10, 15)
        self.game_window.blit(score_surface, score_rect)
        
        pygame.display.update()
        self.fps_controller.tick(self.difficulty)
    
    def close(self):
        """Close the game window."""
        if self.display:
            pygame.quit()
