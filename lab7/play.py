"""
Play Snake Game with Trained RL Agent
Loads a trained model and plays the game.
"""

import os
from snake_game import SnakeGame
from rl_agent import DQNAgent


def play_with_agent(model_path='models/snake_dqn_final.pth', num_games=5):
    """
    Play the game with a trained agent.
    
    Args:
        model_path: Path to the trained model
        num_games: Number of games to play
    """
    # Initialize game (with display enabled)
    game = SnakeGame(display=True, difficulty=120)
    
    # Initialize agent
    agent = DQNAgent(
        state_size=11,
        action_size=3,
        epsilon=0.0  # No exploration during play
    )
    
    # Load model
    if not agent.load(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the agent first using train.py")
        return
    
    print(f"Playing {num_games} games with trained agent...")
    print("-" * 50)
    
    scores = []
    
    for game_num in range(num_games):
        state = game.reset()
        done = False
        steps = 0
        
        while not done:
            # Get action from agent (no exploration)
            action = agent.act(state, training=False)
            
            # Take step
            next_state, reward, done, info = game.step(action)
            
            # Render game
            game.render()
            
            state = next_state
            steps += 1
        
        score = info.get('score', 0)
        scores.append(score)
        print(f"Game {game_num + 1}: Score = {score}, Steps = {steps}")
    
    print("-" * 50)
    print(f"Average Score: {sum(scores) / len(scores):.2f}")
    print(f"Best Score: {max(scores)}")
    print(f"Worst Score: {min(scores)}")
    
    game.close()


if __name__ == '__main__':
    # Check if model exists
    model_path = 'models/snake_dqn_final.pth'
    
    if not os.path.exists(model_path):
        print("No trained model found. Please run train.py first.")
    else:
        play_with_agent(model_path=model_path, num_games=5)
