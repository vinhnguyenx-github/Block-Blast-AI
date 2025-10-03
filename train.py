"""
Training script for Block Blast Deep RL Agent
Run this to train the agent on the game.
"""

import numpy as np
from grid import Grid
from piece import Piece
from gameLogic import clear_lines, can_place_piece, can_place_any, generate_solvable_pieces
from rl_agent import BlockBlastAgent, Experience
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


class BlockBlastEnv:
    """Game environment wrapper for RL training."""
    
    def __init__(self):
        self.board = Grid(8, 8, 60)
        self.pieces = []
        self.score = 0
        self.total_moves = 0
        
    def reset(self):
        """Reset environment to initial state."""
        self.board.clear()
        self.pieces = generate_solvable_pieces(self.board)
        self.score = 0
        self.total_moves = 0
        return self._get_state()
        
    def _get_state(self):
        """Get current state."""
        return (self.board.grid.copy(), self.pieces.copy())
        
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info).
        action: (piece_idx, row, col)
        """
        if action is None:
            return self._get_state(), -100, True, {'reason': 'no_valid_action'}
            
        piece_idx, row, col = action
        piece = self.pieces[piece_idx]
        
        if not can_place_piece(self.board, piece, row, col):
            return self._get_state(), -50, True, {'reason': 'invalid_placement'}
            
        # Store old board state for reward calculation
        old_cells = np.sum(self.board.grid)
        
        # Place piece
        for r in range(piece.rows):
            for c in range(piece.cols):
                if piece.grid[r, c] == 1:
                    self.board.set_cell(row + r, col + c, 1)
                    
        # Clear lines and calculate reward
        cleared = clear_lines(self.board)
        self.score += cleared
        self.total_moves += 1
        
        # Calculate reward
        new_cells = np.sum(self.board.grid)
        placed_cells = np.sum(piece.grid)
        actual_cleared = old_cells + placed_cells - new_cells
        
        reward = actual_cleared * 2 + 1  # 2 points per cleared, 1 for valid move
        
        # Penalize board density
        density = new_cells / 64.0
        reward -= density * 5
        
        # Bonus for clearing multiple lines
        if actual_cleared > 8:
            reward += (actual_cleared - 8) * 3
            
        # Remove used piece
        self.pieces[piece_idx] = None
        
        # Generate new pieces if all used
        if all(p is None for p in self.pieces):
            self.pieces = generate_solvable_pieces(self.board)
            reward += 10  # Bonus for using all pieces
            
        # Check game over
        done = not can_place_any(self.board, self.pieces)
        if done:
            reward = -100
            
        info = {
            'score': self.score,
            'moves': self.total_moves,
            'cleared': actual_cleared,
            'density': density
        }
        
        return self._get_state(), reward, done, info
        
    def get_valid_actions(self):
        """Get all valid actions in current state."""
        valid_actions = []
        
        for piece_idx, piece in enumerate(self.pieces):
            if piece is None:
                continue
                
            for row in range(8):
                for col in range(8):
                    if can_place_piece(self.board, piece, row, col):
                        valid_actions.append((piece_idx, row, col))
                        
        return valid_actions


def train_agent(episodes=10000, save_freq=500, render_freq=100):
    """Train the Block Blast agent."""
    
    env = BlockBlastEnv()
    agent = BlockBlastAgent(
        board_size=8,
        lr=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        buffer_size=50000,
        batch_size=128
    )
    
    # Training metrics
    episode_rewards = []
    episode_scores = []
    episode_moves = []
    losses = []
    
    best_score = 0
    
    print("Starting training...")
    print(f"Device: {agent.device}")
    
    for episode in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                done = True
                break
                
            # Select action
            board, pieces = state
            action = agent.select_action(board, pieces, valid_actions, train=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Store experience
            agent.memory.add(Experience(state, action, reward, next_state, done))
            
            # Train
            if len(agent.memory) >= agent.batch_size:
                beta = min(1.0, 0.4 + episode * 0.00006)  # Anneal beta
                loss = agent.train_step(beta=beta)
                if loss is not None:
                    episode_loss.append(loss)
                    
            state = next_state
            agent.steps += 1
            
            # Update target network
            if agent.steps % 1000 == 0:
                agent.update_target_network()
                
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_scores.append(info['score'])
        episode_moves.append(info['moves'])
        if episode_loss:
            losses.append(np.mean(episode_loss))
            
        # Track best score
        if info['score'] > best_score:
            best_score = info['score']
            agent.save('best_model.pth')
            
        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_score = np.mean(episode_scores[-50:])
            avg_moves = np.mean(episode_moves[-50:])
            avg_loss = np.mean(losses[-50:]) if losses[-50:] else 0
            
            print(f"\nEpisode {episode+1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Score: {avg_score:.2f}")
            print(f"  Avg Moves: {avg_moves:.2f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Best Score: {best_score}")
            
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(f'checkpoint_ep{episode+1}.pth')
            
            # Save training metrics
            with open(f'training_metrics_ep{episode+1}.pkl', 'wb') as f:
                pickle.dump({
                    'rewards': episode_rewards,
                    'scores': episode_scores,
                    'moves': episode_moves,
                    'losses': losses
                }, f)
                
    # Save final model
    agent.save('final_model.pth')
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_scores, episode_moves, losses)
    
    return agent, episode_rewards, episode_scores


def plot_training_curves(rewards, scores, moves, losses):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Smooth curves
    window = 50
    
    # Episode rewards
    axes[0, 0].plot(rewards, alpha=0.3, label='Raw')
    if len(rewards) >= window:
        smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards)), smooth_rewards, label='Smoothed')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Game scores
    axes[0, 1].plot(scores, alpha=0.3, label='Raw')
    if len(scores) >= window:
        smooth_scores = np.convolve(scores, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(scores)), smooth_scores, label='Smoothed')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Game Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Moves per episode
    axes[1, 0].plot(moves, alpha=0.3, label='Raw')
    if len(moves) >= window:
        smooth_moves = np.convolve(moves, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(moves)), smooth_moves, label='Smoothed')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Moves')
    axes[1, 0].set_title('Moves per Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Training loss
    if losses:
        axes[1, 1].plot(losses, alpha=0.3, label='Raw')
        if len(losses) >= window:
            smooth_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(losses)), smooth_loss, label='Smoothed')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("Training curves saved to 'training_curves.png'")
    plt.close()


if __name__ == '__main__':
    # Train the agent
    agent, rewards, scores = train_agent(
        episodes=10000,
        save_freq=500,
        render_freq=100
    )
    
    print("\nTraining complete!")
    print(f"Final average score (last 100 episodes): {np.mean(scores[-100:]):.2f}")
    print(f"Best score achieved: {max(scores)}")