"""
Snake Eater in PyGame originally created by:
Rajat Dipta Biswas
https://github.com/rajatdiptabiswas/snake-pygame

Changes in the code were made for the implementation of fuzzy logic
Fuzzy logic implemented by:
Kamil Suchomski
Kamil Koniak

Problem with Snake Eater: - 

- The difficulty level is set to 120 (impossible)
- Game is requiring fast decision-making and immediate reactions (Up, Down, Left, or Right).
- The snake must avoid colliding with walls.
- The snake must collect red apples, the positions of which are randomly generated.
- The snake must avoid colliding with its own body.
- The biggest difficulty is that the snake grows one length each time it eats an apple, increasing the likelihood of colliding with itself. 
"""

import pygame, sys, time, random
from fuzzy_logic import make_fuzzy_decision
import threading
from functools import lru_cache


# Difficulty settings
# Easy      ->  10
# Medium    ->  25
# Hard      ->  40
# Harder    ->  60
# Impossible->  120
difficulty = 120

# Window size
frame_size_x = 720
frame_size_y = 480

# Checks for errors encountered
check_errors = pygame.init()
# pygame.init() example output -> (6, 0)
# second number in tuple gives number of errors
if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')


# Initialise game window
pygame.display.set_caption('Snake Eater')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))


# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)


# FPS (frames per second) controller
fps_controller = pygame.time.Clock()

def get_quarter(x, y):
    """
    Determine which quarter of the game window a given position belongs to.
    
    Args:
        x: X coordinate of the position
        y: Y coordinate of the position
    
    Returns:
        int: Quarter number (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
    """
    mid_x = frame_size_x // 2
    mid_y = frame_size_y // 2
    
    if x < mid_x and y < mid_y:
        return 0  # Top-left
    elif x >= mid_x and y < mid_y:
        return 1  # Top-right
    elif x < mid_x and y >= mid_y:
        return 2  # Bottom-left
    else:
        return 3  # Bottom-right

@lru_cache(maxsize=1)
def get_all_valid_positions():
    """
    Pre-compute all valid food positions in the game window, avoiding walls.
    
    Returns:
        list: List of valid (x, y) tuple positions where food can spawn
    """
    valid_positions = []
    for x in range(2, (frame_size_x//10)-1):
        for y in range(2, (frame_size_y//10)-1):
            valid_positions.append((x * 10, y * 10))
    return valid_positions

@lru_cache(maxsize=64)
def get_snake_conflict_sets(snake_body_tuple):
    """
    Get sets for efficient conflict detection with snake body positions.
    
    Args:
        snake_body_tuple: Tuple of tuples representing snake body segment positions
    
    Returns:
        tuple: Three sets - occupied_positions (all body positions), 
               occupied_x (all x coordinates), occupied_y (all y coordinates)
    """
    occupied_positions = set()
    occupied_x = set()
    occupied_y = set()
    
    for segment in snake_body_tuple:
        occupied_positions.add(segment)
        occupied_x.add(segment[0])
        occupied_y.add(segment[1])
    
    return occupied_positions, occupied_x, occupied_y

@lru_cache(maxsize=128)
def get_quarter_cached(x, y):
    """
    Cached version of get_quarter function for performance optimization.
    
    Args:
        x: X coordinate of the position
        y: Y coordinate of the position
    
    Returns:
        int: Quarter number (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
    """
    mid_x = frame_size_x // 2
    mid_y = frame_size_y // 2
    
    if x < mid_x and y < mid_y:
        return 0  # Top-left
    elif x >= mid_x and y < mid_y:
        return 1  # Top-right
    elif x < mid_x and y >= mid_y:
        return 2  # Bottom-left
    else:
        return 3  # Bottom-right

def clear_food_generation_cache():
    """Clear all LRU caches related to food generation to free memory if needed."""
    get_all_valid_positions.cache_clear()
    get_snake_conflict_sets.cache_clear()
    get_quarter_cached.cache_clear()

# Generate valid food position (not on snake body, not in same quarter as snake head)
def generate_food_position():
    """
    Generate a random valid food position that doesn't conflict with the snake body
    and prefers positions in a different quarter than the snake head.
    
    Returns:
        list: [x, y] coordinates of a valid food position
    """
    # Get pre-computed valid positions
    valid_positions = get_all_valid_positions()
    
    # Convert snake body to tuple for caching
    snake_body_tuple = tuple(tuple(segment) for segment in snake_body)
    
    # Get cached conflict sets
    occupied_positions, occupied_x, occupied_y = get_snake_conflict_sets(snake_body_tuple)
    
    # Get snake head quarter
    snake_head_quarter = get_quarter_cached(snake_pos[0], snake_pos[1])
    
    # Filter valid positions that don't conflict with snake
    available_positions = []
    for pos in valid_positions:
        food_x, food_y = pos
        
        # Check if position conflicts with snake body
        if (pos not in occupied_positions and 
            food_x not in occupied_x and 
            food_y not in occupied_y):
            
            # Check if food is in different quarter than snake head
            food_quarter = get_quarter_cached(food_x, food_y)
            if food_quarter != snake_head_quarter:
                available_positions.append([food_x, food_y])
    
    # If no positions available, fallback to any non-conflicting position
    if not available_positions:
        for pos in valid_positions:
            food_x, food_y = pos
            if (pos not in occupied_positions and 
                food_x not in occupied_x and 
                food_y not in occupied_y):
                available_positions.append([food_x, food_y])
    
    # Randomly select from available positions
    if available_positions:
        return random.choice(available_positions)
    else:
        # Emergency fallback - return any valid position
        return list(random.choice(valid_positions))


# Game variables
snake_pos = [100, 50]
snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]

food_pos = generate_food_position()
food_spawn = True

direction = 'RIGHT'
change_to = direction

score = 0


# Check for new best score
def check_best_score():
    """
    Check if current score is a new best score and update the best_score.txt file if so.
    
    Returns:
        bool: True if this is a new best score, False otherwise
    """
    try:
        with open('best_score.txt', 'r') as f:
            best_score = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        best_score = 0
    
    is_new_best = score > best_score
    
    # Update best score if new record
    if is_new_best:
        with open('best_score.txt', 'w') as f:
            f.write(str(score))
    
    return is_new_best


# Game Over
def game_over():
    """
    Display the game over screen, check for new best score, and exit the game.
    Shows the final score and a "New Best Score!" message if applicable.
    """
    # Check for new best score
    is_new_best = check_best_score()
    
    # Display game over screen
    my_font = pygame.font.SysFont('times new roman', 90)
    game_over_surface = my_font.render('YOU DIED', True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (frame_size_x/2, frame_size_y/4)
    game_window.fill(black)
    game_window.blit(game_over_surface, game_over_rect)
    
    # Show current score
    show_score(0, red, 'times', 20)
    
    # Show new best score message if applicable
    if is_new_best:
        best_font = pygame.font.SysFont('times new roman', 40)
        best_surface = best_font.render(f'New Best Score! {score}', True, green)
        best_rect = best_surface.get_rect()
        best_rect.midtop = (frame_size_x/2, frame_size_y/2)
        game_window.blit(best_surface, best_rect)
    
    pygame.display.flip()
    time.sleep(3)
    pygame.quit()
    sys.exit()


# Score
def show_score(choice, color, font, size):
    """
    Display the current score on the game screen.
    
    Args:
        choice: Display position (1=top-left during game, 0=center during game over)
        color: RGB tuple color for the score text
        font: Font name string
        size: Font size integer
    """
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    if choice == 1:
        score_rect.midtop = (frame_size_x/10, 15)
    else:
        score_rect.midtop = (frame_size_x/2, frame_size_y/1.25)
    game_window.blit(score_surface, score_rect)
    # pygame.display.flip()


# Main logic
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # Whenever a key is pressed down
        elif event.type == pygame.KEYDOWN:
            # Esc -> Create event to quit the game
            if event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))

    # Add timeout protection for decision making
    decision_result = [None]
    decision_error = [None]
    
    def make_decision_thread():
        try:
            decision_result[0] = make_fuzzy_decision(frame_size_x, frame_size_y, snake_pos, food_pos, snake_body, direction)
        except Exception as e:
            decision_error[0] = e
    
    # Start decision thread with timeout
    decision_thread = threading.Thread(target=make_decision_thread)
    decision_thread.daemon = True
    decision_thread.start()
    decision_thread.join(timeout=1.0)  # 1 second timeout
    
    if decision_thread.is_alive():
        print(f"TIMEOUT: Decision took too long! Snake length: {len(snake_body)}")
        print("Using current direction as fallback...")
        change_to = direction
    elif decision_error[0]:
        print(f"Decision error: {decision_error[0]}")
        change_to = direction
    else:
        change_to = decision_result[0]

    # Making sure the snake cannot move in the opposite direction instantaneously
    if change_to == 'UP' and direction != 'DOWN':
        direction = 'UP'
    if change_to == 'DOWN' and direction != 'UP':
        direction = 'DOWN'
    if change_to == 'LEFT' and direction != 'RIGHT':
        direction = 'LEFT'
    if change_to == 'RIGHT' and direction != 'LEFT':
        direction = 'RIGHT'



    # Moving the snake
    if direction == 'UP':
        snake_pos[1] -= 10
    if direction == 'DOWN':
        snake_pos[1] += 10
    if direction == 'LEFT':
        snake_pos[0] -= 10
    if direction == 'RIGHT':
        snake_pos[0] += 10

    # Snake body growing mechanism
    snake_body.insert(0, list(snake_pos))
    if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
        score += 1
        food_spawn = False
    else:
        snake_body.pop()

    # Spawning food on the screen
    if not food_spawn:
        food_pos = generate_food_position()
    food_spawn = True

    # GFX
    game_window.fill(black)
    for pos in snake_body:
        # Snake body
        # .draw.rect(play_surface, color, xy-coordinate)
        # xy-coordinate -> .Rect(x, y, size_x, size_y)
        pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

    # Snake food
    pygame.draw.rect(game_window, red, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

    # Game Over conditions
    # Getting out of bounds
    if snake_pos[0] < 0 or snake_pos[0] > frame_size_x-10:
        game_over()
    if snake_pos[1] < 0 or snake_pos[1] > frame_size_y-10:
        game_over()
    # Touching the snake body
    for block in snake_body[1:]:
        if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
            game_over()

    show_score(1, white, 'consolas', 20)
    # Refresh game screen
    pygame.display.update()
    # Refresh rate
    fps_controller.tick(difficulty)