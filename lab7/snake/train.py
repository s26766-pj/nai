"""
Autors:
Kamil Suchomski s21974
Kamil Koniak s26766

Training script for RL Snake Game
Trains a DQN agent to play Snake using reinforcement learning.

Based on research from:
- Nancy Zhou's Medium article on RL Snake
- Reinforcement Learning Tutorial (ResearchGate)
"""

import os
import matplotlib.pyplot as plt
from snake_game import SnakeGame
from rl_agent import DQNAgent
import numpy as np


def train_agent(num_episodes=2000, save_interval=1000, model_dir='models'):
    """
    Train the DQN agent.
    
    Args:
        num_episodes: Number of training episodes
        save_interval: Save model every N episodes
        model_dir: Directory to save models
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize game and agent
    # Set display=False for faster training (set to True to watch training)
    game = SnakeGame(display=False, difficulty=120)
    agent = DQNAgent(
        state_size=11,
        action_size=3,
        lr=0.001,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=100000,
        batch_size=64
    )
    
    # Training statistics
    scores = []
    avg_scores = []
    
    print("Starting training...")
    print(f"Training for {num_episodes} games (episodes)")
    print("Note: Each episode = one complete game (from start to game over)")
    print("-" * 50)
    
    for episode in range(num_episodes):
        state = game.reset()
        steps = 0
        done = False
        
        while not done:
            # Choose action
            action = agent.act(state, training=True)
            
            # Take step
            next_state, reward, done, info = game.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            state = next_state
            steps += 1
        
        # Record score
        score = info.get('score', 0)
        scores.append(score)
        
        # Calculate average score
        if len(scores) >= 100:
            avg_score = np.mean(scores[-100:])
        else:
            avg_score = np.mean(scores)
        avg_scores.append(avg_score)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Game {episode + 1}/{num_episodes} (Episode {episode + 1}) | "
                  f"Score: {score} | "
                  f"Avg Score (last 100 games): {avg_score:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Steps: {steps}")
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(model_dir, f'snake_dqn_episode_{episode + 1}.pth')
            agent.save(model_path)
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'snake_dqn_final.pth')
    agent.save(final_model_path)
    
    # Plot training progress
    plot_training_progress(scores, avg_scores, save_path='training_progress.png')
    
    print("\nTraining completed!")
    print(f"Final average score (last 100 games): {np.mean(scores[-100:]):.2f}")
    print(f"Best score: {max(scores)}")
    print(f"Total games played: {num_episodes}")
    
    return agent, scores, avg_scores


def plot_training_progress(scores, avg_scores, save_path='training_progress.png'):
    """
    Plot training progress.
    
    Args:
        scores: List of scores per game
        avg_scores: List of average scores (rolling average)
        save_path: Path to save the plot
    """
    plt.figure(figsize=(14, 6))
    
    # Create x-axis (number of games)
    num_games = len(scores)
    games = list(range(1, num_games + 1))
    
    # Plot 1: Score per game and rolling average
    plt.subplot(1, 2, 1)
    plt.plot(games, scores, alpha=0.4, color='blue', label='Score per Game', linewidth=0.5)
    plt.plot(games, avg_scores, color='red', label='Mean Score (rolling 100 games)', linewidth=2)
    plt.xlabel('Number of Games', fontsize=11)
    plt.ylabel('Score', fontsize=11)
    plt.title('Training Progress: Score vs Number of Games', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Mean score over time (cleaner view)
    plt.subplot(1, 2, 2)
    plt.plot(games, avg_scores, color='green', label='Mean Score (rolling 100 games)', linewidth=2.5)
    plt.xlabel('Number of Games', fontsize=11)
    plt.ylabel('Mean Score', fontsize=11)
    plt.title('Mean Score Over Training', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    final_mean = avg_scores[-1] if avg_scores else 0
    best_score = max(scores) if scores else 0
    stats_text = f'Final Mean: {final_mean:.2f}\nBest Score: {best_score}'
    bbox_props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=bbox_props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training plot saved to {save_path}")
    print("  - X-axis: Number of Games")
    print("  - Y-axis: Score / Mean Score")




if __name__ == '__main__':
    # Train the agent
    agent, scores, avg_scores = train_agent(
        num_episodes=2000,
        save_interval=1000,
        model_dir='models'
    )
