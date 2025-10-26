"""
Fuzzy Logic System for Snake Game AI
Authors: Kamil Suchomski, Kamil Koniak

This module contains the fuzzy logic implementation for decision making in the Snake game.
"""

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=1)
def setup_fuzzy_system():
    """
    Setup and return the fuzzy control system for snake decision making.
    
    This function initializes a fuzzy logic control system with:
    - 9 input variables (wall and body distances in 3 directions, food attraction in 3 directions)
    - 1 output variable (turn decision from -100 to 100)
    - 14 fuzzy rules prioritizing safety (avoiding walls and body) and food seeking
    
    Returns:
        ctrl.ControlSystemSimulation: The configured fuzzy control simulation system
    """

    # Input 1: Wall distance forward
    wall_forward = ctrl.Antecedent(np.arange(-10, 101, 1), 'wall_forward')
    wall_forward['danger'] = fuzz.trimf(wall_forward.universe, [-10, -4, 2])
    wall_forward['safe'] = fuzz.trapmf(wall_forward.universe, [1, 50, 100, 100])

    # Input 2: Wall distance left
    wall_left = ctrl.Antecedent(np.arange(-10, 101, 1), 'wall_left')
    wall_left['danger'] = fuzz.trimf(wall_left.universe, [-10, -4, 2])
    wall_left['safe'] = fuzz.trapmf(wall_left.universe, [1, 50, 100, 100])

    # Input 3: Wall distance right
    wall_right = ctrl.Antecedent(np.arange(-10, 101, 1), 'wall_right')
    wall_right['danger'] = fuzz.trimf(wall_right.universe, [-10, -4, 2])
    wall_right['safe'] = fuzz.trapmf(wall_right.universe, [1, 50, 100, 100])

    # Input 4: Body distance forward
    body_forward = ctrl.Antecedent(np.arange(-1, 101, 1), 'body_forward')
    body_forward['danger'] = fuzz.trimf(body_forward.universe, [-1, 4, 5])
    body_forward['safe'] = fuzz.trapmf(body_forward.universe, [6, 25, 100, 100])

    # Input 5: Body distance left
    body_left = ctrl.Antecedent(np.arange(-1, 101, 1), 'body_left')
    body_left['danger'] = fuzz.trimf(body_left.universe, [-1, 1, 3])
    body_left['safe'] = fuzz.trapmf(body_left.universe, [3, 25, 100, 100])

    # Input 6: Body distance right
    body_right = ctrl.Antecedent(np.arange(-1, 101, 1), 'body_right')
    body_right['danger'] = fuzz.trimf(body_right.universe, [-1, 1, 3])
    body_right['safe'] = fuzz.trapmf(body_right.universe, [3, 25, 100, 100])

    # Input 7: Food forward attraction
    food_forward = ctrl.Antecedent(np.arange(0, 101, 1), 'food_forward')
    food_forward['far'] = fuzz.trimf(food_forward.universe, [0, 1, 10])
    food_forward['medium'] = fuzz.trimf(food_forward.universe, [11, 35, 50])
    food_forward['close'] = fuzz.trapmf(food_forward.universe, [45, 65, 100, 100])

    # Input 8: Food left attraction
    food_left = ctrl.Antecedent(np.arange(0, 101, 1), 'food_left')
    food_left['far'] = fuzz.trimf(food_left.universe, [0, 1, 10])
    food_left['medium'] = fuzz.trimf(food_left.universe, [11, 35, 50])
    food_left['close'] = fuzz.trapmf(food_left.universe, [45, 65, 100, 100])

    # Input 9: Food right attraction
    food_right = ctrl.Antecedent(np.arange(0, 101, 1), 'food_right')
    food_right['far'] = fuzz.trimf(food_right.universe, [0, 1, 10])
    food_right['medium'] = fuzz.trimf(food_right.universe, [11, 35, 50])
    food_right['close'] = fuzz.trapmf(food_right.universe, [45, 65, 100, 100])

    # Output: Turn decision (-100 to 100)
    turn_decision = ctrl.Consequent(np.arange(-100, 101, 1), 'turn_decision')
    turn_decision['turn_left_strong'] = fuzz.trapmf(turn_decision.universe, [-100, -100, -70, -40])
    turn_decision['turn_left'] = fuzz.trimf(turn_decision.universe, [-60, -30, 0])
    turn_decision['go_straight'] = fuzz.trimf(turn_decision.universe, [-15, 0, 15])
    turn_decision['turn_right'] = fuzz.trimf(turn_decision.universe, [0, 30, 60])
    turn_decision['turn_right_strong'] = fuzz.trapmf(turn_decision.universe, [40, 70, 100, 100])
    turn_decision.defuzzify_method = 'centroid'
    
    # PRIORITY 1: Critical wall danger ahead - MUST turn to safer side
    rule1 = ctrl.Rule(wall_forward['danger'] & wall_left['safe'] & body_forward['safe'], 
                      turn_decision['turn_left'])
    rule2 = ctrl.Rule(wall_forward['danger'] & wall_right['safe'] & body_forward['safe'], 
                      turn_decision['turn_right'])
    
    # PRIORITY 2: Critical body danger ahead - MUST turn to safer side
    rule3 = ctrl.Rule(body_forward['danger'] & body_left['safe'] & wall_forward['safe'], 
                      turn_decision['turn_left_strong'])
    rule4 = ctrl.Rule(body_forward['danger'] & body_right['safe'] & wall_forward['safe'], 
                      turn_decision['turn_right_strong'])
    
    # PRIORITY 3: Combined wall + body danger - emergency turn
    rule5 = ctrl.Rule(wall_forward['danger'] & body_forward['danger'], 
                      turn_decision['turn_left_strong'])
    
    # PRIORITY 4: Food seeking when safe from both wall and body
    rule6 = ctrl.Rule(food_left['close'] & wall_left['safe'] & body_left['safe'], 
                      turn_decision['turn_left_strong'])
    rule7 = ctrl.Rule(food_right['close'] & wall_right['safe'] & body_right['safe'], 
                      turn_decision['turn_right_strong'])
    rule8 = ctrl.Rule(food_forward['close'] & wall_forward['safe'] & body_forward['safe'], 
                      turn_decision['go_straight'])
    
    # PRIORITY 5: Food seeking - MEDIUM distance
    rule9 = ctrl.Rule(food_left['medium'] & wall_left['safe'] & body_left['safe'], 
                      turn_decision['turn_left'])
    rule10 = ctrl.Rule(food_right['medium'] & wall_right['safe'] & body_right['safe'], 
                       turn_decision['turn_right'])
    rule11 = ctrl.Rule(food_forward['medium'] & wall_forward['safe'] & body_forward['safe'], 
                       turn_decision['go_straight'])
    
    # PRIORITY 6: Avoid side dangers (weaker - only when no food nearby)
    rule12 = ctrl.Rule((wall_left['danger'] | body_left['danger']) & ~food_right['close'] & ~food_right['medium'], 
                       turn_decision['turn_right'])
    rule13 = ctrl.Rule((wall_right['danger'] | body_right['danger']) & ~food_left['close'] & ~food_left['medium'], 
                       turn_decision['turn_left'])
    
    # PRIORITY 7: Default - keep going if safe and no food nearby
    rule14 = ctrl.Rule(wall_forward['safe'] & body_forward['safe'] & ~food_left['close'] & ~food_right['close'] & ~food_left['medium'] & ~food_right['medium'], 
                       turn_decision['go_straight'])

    # Create control system
    control_system = ctrl.ControlSystem([
        rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14
    ])

    # Create simulation
    simulation = ctrl.ControlSystemSimulation(control_system)
    
    return simulation


def make_fuzzy_decision(frame_size_x, frame_size_y, snake_pos, food_pos, snake_body, current_direction):
    """
    Uses fuzzy logic to decide the next move for the snake.
    
    Analyzes the game state (wall distances, body distances, food position) and uses
    fuzzy inference to determine the best direction to move. Prioritizes safety
    (avoiding walls and body) while seeking food when safe.
    
    Args:
        frame_size_x: Width of the game window in pixels
        frame_size_y: Height of the game window in pixels
        snake_pos: List [x, y] coordinates of snake head
        food_pos: List [x, y] coordinates of food
        snake_body: List of [x, y] lists representing all snake segments
        current_direction: Current direction ('UP', 'DOWN', 'LEFT', 'RIGHT')
    
    Returns:
        str: Next direction ('UP', 'DOWN', 'LEFT', or 'RIGHT'). 
             Returns current_direction as fallback on error.
    """
    
    try:
        # Setup fuzzy system (cached)
        simulation = setup_fuzzy_system()
        
        # Convert lists to tuples for caching
        snake_pos_tuple = tuple(snake_pos)
        food_pos_tuple = tuple(food_pos)
        snake_body_tuple = tuple(tuple(segment) for segment in snake_body)
        
        # Step 1: Get sensor data (0-100 normalized) - all cached
        wall_distances = get_relative_wall_distances(frame_size_x, frame_size_y, snake_pos_tuple, current_direction)
        body_distances = get_relative_body_distances(frame_size_x, frame_size_y, snake_pos_tuple, snake_body_tuple, current_direction)
        food_position = get_relative_food_position(frame_size_x, frame_size_y, snake_pos_tuple, food_pos_tuple, current_direction)
        
        # Step 2: Feed separate inputs into fuzzy system
        simulation.input['wall_forward'] = wall_distances['forward']
        simulation.input['wall_left'] = wall_distances['left']
        simulation.input['wall_right'] = wall_distances['right']
        simulation.input['body_forward'] = body_distances['forward']
        simulation.input['body_left'] = body_distances['left']
        simulation.input['body_right'] = body_distances['right']
        simulation.input['food_forward'] = food_position['forward']
        simulation.input['food_left'] = food_position['left']
        simulation.input['food_right'] = food_position['right']
    

        # Step 3: Compute fuzzy output
        try:
            simulation.compute()
        except Exception as compute_error:
            print(f"Fuzzy computation error: {compute_error}")
            print(f"Snake length: {len(snake_body)}")
            return current_direction  # Fallback to current direction
            
        # Step 4: Get decision value (-100 to 100)
        decision_value = simulation.output['turn_decision']
        
        # Step 5: Convert fuzzy output to actual direction
        if decision_value < -1:
            # Turn left (negative values)
            return turn_left(current_direction)
        elif decision_value > 1:
            # Turn right (positive values)
            return turn_right(current_direction)
        else:
            # Go straight (values near zero)
            return current_direction

    except (KeyError) as e:
        print(f"Warning!")
        return current_direction
    
    except (ValueError) as e:
        # No rules fired or computation failed, use fallback logic
        print(f"Warning: Fuzzy system failed ({e}), using fallback logic")
        return current_direction
    
    except Exception as e:
        print(f"Unexpected error in fuzzy decision: {e}")
        return current_direction


def turn_left(current_direction):
    """
    Returns the direction after turning 90° left (counter-clockwise).
    
    Args:
        current_direction: Current snake direction ('UP', 'DOWN', 'LEFT', 'RIGHT')
    
    Returns:
        str: New direction after turning left, or same direction if invalid input
    """
    turns = {
        'UP': 'LEFT',
        'LEFT': 'DOWN',
        'DOWN': 'RIGHT',
        'RIGHT': 'UP'
    }
    return turns.get(current_direction, current_direction)


def turn_right(current_direction):
    """
    Returns the direction after turning 90° right (clockwise).
    
    Args:
        current_direction: Current snake direction ('UP', 'DOWN', 'LEFT', 'RIGHT')
    
    Returns:
        str: New direction after turning right, or same direction if invalid input
    """
    turns = {
        'UP': 'RIGHT',
        'RIGHT': 'DOWN',
        'DOWN': 'LEFT',
        'LEFT': 'UP'
    }
    return turns.get(current_direction, current_direction)


@lru_cache(maxsize=128)
def get_relative_wall_distances(frame_size_x, frame_size_y, snake_pos_tuple, current_direction):
    """
    Returns distances to walls relative to snake's current direction.
    Snake can "look" forward, left, and right from its perspective.
    
    Args:
        frame_size_x: Width of game window (default 720)
        frame_size_y: Height of game window (default 480)
        snake_pos_tuple: (x, y) position of snake head (tuple for caching)
        current_direction: 'UP', 'DOWN', 'LEFT', 'RIGHT'

    
    Returns:
        dict: {
            'forward': normalized distance (0-100), 0=at wall, 100=far from wall
            'left': normalized distance (0-100), 0=at wall, 100=far from wall
            'right': normalized distance (0-100), 0=at wall, 100=far from wall
        }
    """
    x, y = snake_pos_tuple
    
    # Calculate absolute distances to all walls in pixels
    distance_to_top = y
    distance_to_bottom = (frame_size_y - 10) - y
    distance_to_left_wall = x
    distance_to_right_wall = (frame_size_x - 10) - x
    
    # Maximum possible distances for normalization
    max_x = frame_size_x - 10
    max_y = frame_size_y - 10
    
    # Map relative directions based on current heading
    if current_direction == 'UP':
        forward = distance_to_top
        left = distance_to_left_wall
        right = distance_to_right_wall
        max_forward = max_y
        max_left = max_x
        max_right = max_x
    
    elif current_direction == 'DOWN':
        forward = distance_to_bottom
        left = distance_to_right_wall      # Left from snake's view = right wall
        right = distance_to_left_wall      # Right from snake's view = left wall
        max_forward = max_y
        max_left = max_x
        max_right = max_x
    
    elif current_direction == 'LEFT':
        forward = distance_to_left_wall
        left = distance_to_bottom
        right = distance_to_top
        max_forward = max_x
        max_left = max_y
        max_right = max_y
    
    elif current_direction == 'RIGHT':
        forward = distance_to_right_wall
        left = distance_to_top
        right = distance_to_bottom
        max_forward = max_x
        max_left = max_y
        max_right = max_y
    
    else:
        # Default case (shouldn't happen)
        forward = left = right = 0
        max_forward = max_left = max_right = 1
    
    # Normalize to 0-100 (0=at wall, 100=far from wall)
    forward_normalized = int((forward / max_forward) * 100) if max_forward > 0 else 0
    left_normalized = int((left / max_left) * 100) if max_left > 0 else 0
    right_normalized = int((right / max_right) * 100) if max_right > 0 else 0
    
    return {
        'forward': forward_normalized,
        'left': left_normalized,
        'right': right_normalized
    }

@lru_cache(maxsize=64)
def get_relative_body_distances(frame_size_x, frame_size_y, snake_pos_tuple, snake_body_tuple, current_direction):
    """
    Returns distances to snake's own body relative to current direction.
    Snake can "look" forward, left, and right to detect its own body.
    
    Args:
        frame_size_x: Width of game window (default 720)
        frame_size_y: Height of game window (default 480)
        snake_pos_tuple: (x, y) position of snake head (tuple for caching)
        snake_body_tuple: Tuple of (x, y) positions of all snake segments (tuple for caching)
        current_direction: 'UP', 'DOWN', 'LEFT', 'RIGHT'

    
    Returns:
        dict: {
            'forward': normalized distance (0-100), 0=body very close, 100=no body/far
            'left': normalized distance (0-100), 0=body very close, 100=no body/far
            'right': normalized distance (0-100), 0=body very close, 100=no body/far
        }
    """
    x, y = snake_pos_tuple
    
    # Maximum possible distances in each direction
    max_x = frame_size_x - 10
    max_y = frame_size_y - 10
    
    # Determine max distances for normalization based on direction
    if current_direction in ['UP', 'DOWN']:
        max_forward = max_y
        max_left = max_x
        max_right = max_x
    else:  # LEFT or RIGHT
        max_forward = max_x
        max_left = max_y
        max_right = max_y
    
    # Initialize distances to their respective maximums (when no body found)
    distance_forward = max_forward
    distance_left = max_left
    distance_right = max_right
    
    # Skip the head (index 0) and check body segments
    for segment in snake_body_tuple[1:]:
        seg_x, seg_y = segment
        
        if current_direction == 'UP':
            # Forward = UP (decreasing y, same x)
            if seg_x == x and seg_y < y:
                distance_forward = min(distance_forward, y - seg_y)
            # Left = LEFT (decreasing x, same y)
            if seg_y == y and seg_x < x:
                distance_left = min(distance_left, x - seg_x)
            # Right = RIGHT (increasing x, same y)
            if seg_y == y and seg_x > x:
                distance_right = min(distance_right, seg_x - x)
        
        elif current_direction == 'DOWN':
            # Forward = DOWN (increasing y, same x)
            if seg_x == x and seg_y > y:
                distance_forward = min(distance_forward, seg_y - y)
            # Left = RIGHT (increasing x, same y)
            if seg_y == y and seg_x > x:
                distance_left = min(distance_left, seg_x - x)
            # Right = LEFT (decreasing x, same y)
            if seg_y == y and seg_x < x:
                distance_right = min(distance_right, x - seg_x)
        
        elif current_direction == 'LEFT':
            # Forward = LEFT (decreasing x, same y)
            if seg_y == y and seg_x < x:
                distance_forward = min(distance_forward, x - seg_x)
            # Left = DOWN (increasing y, same x)
            if seg_x == x and seg_y > y:
                distance_left = min(distance_left, seg_y - y)
            # Right = UP (decreasing y, same x)
            if seg_x == x and seg_y < y:
                distance_right = min(distance_right, y - seg_y)
        
        elif current_direction == 'RIGHT':
            # Forward = RIGHT (increasing x, same y)
            if seg_y == y and seg_x > x:
                distance_forward = min(distance_forward, seg_x - x)
            # Left = UP (decreasing y, same x)
            if seg_x == x and seg_y < y:
                distance_left = min(distance_left, y - seg_y)
            # Right = DOWN (increasing y, same x)
            if seg_x == x and seg_y > y:
                distance_right = min(distance_right, seg_y - y)
    
    # Normalize to 0-100 (0=body very close, 100=no body/far)
    forward_normalized = int((distance_forward / max_forward) * 100) if max_forward > 0 else 0
    left_normalized = int((distance_left / max_left) * 100) if max_left > 0 else 0
    right_normalized = int((distance_right / max_right) * 100) if max_right > 0 else 0
    
    return {
        'forward': forward_normalized,
        'left': left_normalized,
        'right': right_normalized
    }


@lru_cache(maxsize=128)
def get_relative_food_position(frame_size_x, frame_size_y, snake_pos_tuple, food_pos_tuple, current_direction):
    """
    Analyzes food position relative to snake's current direction.
    Returns normalized distances (0-100) and which direction(s) the food is in.
    
    Args:
        snake_pos_tuple: (x, y) position of snake head (tuple for caching)
        food_pos_tuple: (x, y) position of food (tuple for caching)
        current_direction: 'UP', 'DOWN', 'LEFT', 'RIGHT'
        frame_size_x: Width of game window (default 720)
        frame_size_y: Height of game window (default 480)
    
    Returns:
        dict: {
            'forward': normalized distance (0-100), 0=food very far/not ahead, 100=food close ahead
            'left': normalized distance (0-100), 0=food very far/not left, 100=food close left
            'right': normalized distance (0-100), 0=food very far/not right, 100=food close right
            'total_distance': normalized total distance (0-100), 0=very far, 100=very close
            'is_forward': True if food is ahead,
            'is_left': True if food is to the left,
            'is_right': True if food is to the right
        }
    """
    snake_x, snake_y = snake_pos_tuple
    food_x, food_y = food_pos_tuple
    
    # Calculate absolute differences
    dx = food_x - snake_x  # Positive = food is to the RIGHT
    dy = food_y - snake_y  # Positive = food is DOWN
    
    # Calculate Manhattan distance (total steps to reach food)
    manhattan_distance = abs(dx) + abs(dy)
    
    # Maximum possible Manhattan distance (corner to corner)
    max_manhattan = (frame_size_x - 10) + (frame_size_y - 10)
    
    # Maximum distances for normalization
    max_x = frame_size_x - 10
    max_y = frame_size_y - 10
    
    # Initialize all distances and flags
    forward_dist = 0
    left_dist = 0
    right_dist = 0
    
    is_forward = False
    is_left = False
    is_right = False
    
    # Map relative directions based on current heading
    if current_direction == 'UP':
        # Forward = UP (negative dy)
        if dy < 0:
            forward_dist = abs(dy)
            is_forward = True
        
        # Left = LEFT (negative dx)
        if dx < 0:
            left_dist = abs(dx)
            is_left = True
        # Right = RIGHT (positive dx)
        elif dx > 0:
            right_dist = abs(dx)
            is_right = True
        
        max_forward = max_y
        max_left = max_x
        max_right = max_x
    
    elif current_direction == 'DOWN':
        # Forward = DOWN (positive dy)
        if dy > 0:
            forward_dist = abs(dy)
            is_forward = True
        
        # Left = RIGHT (positive dx)
        if dx > 0:
            left_dist = abs(dx)
            is_left = True
        # Right = LEFT (negative dx)
        elif dx < 0:
            right_dist = abs(dx)
            is_right = True
        
        max_forward = max_y
        max_left = max_x
        max_right = max_x
    
    elif current_direction == 'LEFT':
        # Forward = LEFT (negative dx)
        if dx < 0:
            forward_dist = abs(dx)
            is_forward = True
        
        # Left = DOWN (positive dy)
        if dy > 0:
            left_dist = abs(dy)
            is_left = True
        # Right = UP (negative dy)
        elif dy < 0:
            right_dist = abs(dy)
            is_right = True
        
        max_forward = max_x
        max_left = max_y
        max_right = max_y
    
    elif current_direction == 'RIGHT':
        # Forward = RIGHT (positive dx)
        if dx > 0:
            forward_dist = abs(dx)
            is_forward = True
        
        # Left = UP (negative dy)
        if dy < 0:
            left_dist = abs(dy)
            is_left = True
        # Right = DOWN (positive dy)
        elif dy > 0:
            right_dist = abs(dy)
            is_right = True
        
        max_forward = max_x
        max_left = max_y
        max_right = max_y
    
    else:
        max_forward = max_left = max_right = 1
    
    # Normalize to 0-100, INVERTED: 100=close, 0=far (for consistency with food attraction)
    forward_normalized = int(100 - (forward_dist / max_forward) * 100) if forward_dist > 0 and max_forward > 0 else 0
    left_normalized = int(100 - (left_dist / max_left) * 100) if left_dist > 0 and max_left > 0 else 0
    right_normalized = int(100 - (right_dist / max_right) * 100) if right_dist > 0 and max_right > 0 else 0
    
    # Total distance: 100=very close, 0=very far
    total_normalized = int(100 - (manhattan_distance / max_manhattan) * 100) if max_manhattan > 0 else 0
    
    return {
        'forward': forward_normalized,
        'left': left_normalized,
        'right': right_normalized,
        'total_distance': total_normalized,
        'is_forward': is_forward,
        'is_left': is_left,
        'is_right': is_right
    }

def clear_fuzzy_cache():
    """Clear all LRU caches to free memory if needed."""
    setup_fuzzy_system.cache_clear()
    get_relative_wall_distances.cache_clear()
    get_relative_body_distances.cache_clear()
    get_relative_food_position.cache_clear()


def get_cache_info():
    """Get information about cache usage for debugging."""
    return {
        'fuzzy_system_cache': setup_fuzzy_system.cache_info(),
        'wall_distances_cache': get_relative_wall_distances.cache_info(),
        'body_distances_cache': get_relative_body_distances.cache_info(),
        'food_position_cache': get_relative_food_position.cache_info()
    }