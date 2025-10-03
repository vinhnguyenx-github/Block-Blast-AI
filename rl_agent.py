import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from copy import deepcopy

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNetwork(nn.Module):
    """Deep Q-Network with convolutional layers for board state processing."""
    
    def __init__(self, board_size=8, piece_feature_dim=25, hidden_dim=256):
        super(DQNetwork, self).__init__()
        
        # Convolutional layers for board processing
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate conv output size
        conv_out_size = board_size * board_size * 64
        
        # Piece encoding network
        self.piece_fc1 = nn.Linear(piece_feature_dim * 3, 128)
        self.piece_fc2 = nn.Linear(128, 64)
        
        # Combined processing
        combined_size = conv_out_size + 64
        self.fc1 = nn.Linear(combined_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Q-value for state-action pair
        
    def forward(self, board, pieces):
        """
        Forward pass.
        board: (batch, 1, 8, 8)
        pieces: (batch, 75) - flattened 3 pieces of 5x5 each
        """
        # Process board
        x = F.relu(self.conv1(board))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        # Process pieces
        p = F.relu(self.piece_fc1(pieces))
        p = F.relu(self.piece_fc2(p))
        
        # Combine
        combined = torch.cat([x, p], dim=1)
        h = F.relu(self.fc1(combined))
        h = F.relu(self.fc2(h))
        q_value = self.fc3(h)
        
        return q_value


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer."""
    
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        
    def add(self, experience):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
            
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)


class BlockBlastAgent:
    """Deep RL Agent for Block Blast game."""
    
    def __init__(self, board_size=8, lr=0.0001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=128):
        
        self.board_size = board_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-networks
        self.policy_net = DQNetwork(board_size).to(self.device)
        self.target_net = DQNetwork(board_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(buffer_size)
        
        self.steps = 0
        
    def encode_state(self, board, pieces):
        """Encode board and pieces as tensors."""
        # Board: (8, 8) -> (1, 8, 8)
        board_tensor = torch.FloatTensor(board).unsqueeze(0).to(self.device)
        
        # Pieces: 3 x (5, 5) -> (75,)
        pieces_flat = []
        for piece in pieces:
            if piece is None:
                pieces_flat.append(np.zeros((5, 5)))
            else:
                pieces_flat.append(piece.grid[:5, :5])
        pieces_tensor = torch.FloatTensor(np.concatenate(pieces_flat).flatten()).to(self.device)
        
        return board_tensor, pieces_tensor
        
    def get_valid_actions(self, board, pieces):
        """Get all valid (piece_idx, row, col) placements."""
        valid_actions = []
        
        for piece_idx, piece in enumerate(pieces):
            if piece is None:
                continue
                
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if self._can_place(board, piece, row, col):
                        valid_actions.append((piece_idx, row, col))
                        
        return valid_actions
        
    def _can_place(self, board, piece, row, col):
        """Check if piece can be placed at position."""
        for r in range(5):
            for c in range(5):
                if piece.grid[r, c] == 1:
                    br, bc = row + r, col + c
                    if not (0 <= br < self.board_size and 0 <= bc < self.board_size):
                        return False
                    if board[br, bc] == 1:
                        return False
        return True
        
    def select_action(self, board, pieces, valid_actions, train=True):
        """Select action using epsilon-greedy policy."""
        if train and random.random() < self.epsilon:
            return random.choice(valid_actions) if valid_actions else None
            
        if not valid_actions:
            return None
            
        # Evaluate all valid actions
        board_tensor, pieces_tensor = self.encode_state(board, pieces)
        
        best_action = None
        best_q = float('-inf')
        
        with torch.no_grad():
            for action in valid_actions:
                # Create hypothetical next state
                next_board = self._apply_action(board, pieces, action)
                next_board_tensor = torch.FloatTensor(next_board).unsqueeze(0).unsqueeze(0).to(self.device)
                
                q_value = self.policy_net(next_board_tensor, pieces_tensor.unsqueeze(0))
                
                if q_value.item() > best_q:
                    best_q = q_value.item()
                    best_action = action
                    
        return best_action
        
    def _apply_action(self, board, pieces, action):
        """Apply action and return new board state (with line clears)."""
        piece_idx, row, col = action
        piece = pieces[piece_idx]
        
        new_board = board.copy()
        
        # Place piece
        for r in range(5):
            for c in range(5):
                if piece.grid[r, c] == 1:
                    new_board[row + r, col + c] = 1
                    
        # Clear lines
        new_board = self._clear_lines(new_board)
        
        return new_board
        
    def _clear_lines(self, board):
        """Clear full rows and columns."""
        new_board = board.copy()
        
        # Clear full rows
        full_rows = [r for r in range(self.board_size) if all(new_board[r, :] == 1)]
        for r in full_rows:
            new_board[r, :] = 0
            
        # Clear full columns
        full_cols = [c for c in range(self.board_size) if all(new_board[:, c] == 1)]
        for c in full_cols:
            new_board[:, c] = 0
            
        return new_board
        
    def calculate_reward(self, board, new_board, action, game_over):
        """Calculate reward for the action."""
        if game_over:
            return -100  # Large penalty for losing
            
        # Count cleared squares
        cleared = np.sum(board) - np.sum(new_board) + np.sum(self._get_piece_cells(action))
        
        # Reward for clearing lines
        reward = cleared * 2
        
        # Small reward for valid placement
        reward += 1
        
        # Bonus for board density (encourage keeping board clear)
        density = np.sum(new_board) / (self.board_size ** 2)
        reward -= density * 10
        
        return reward
        
    def _get_piece_cells(self, action):
        """Get number of cells in placed piece."""
        # This is a simplified version; ideally pass pieces to this method
        return 0
        
    def train_step(self, beta=0.4):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample batch
        experiences, indices, weights = self.memory.sample(self.batch_size, beta)
        
        batch_states = []
        batch_pieces = []
        batch_rewards = []
        batch_next_states = []
        batch_next_pieces = []
        batch_dones = []
        
        for exp in experiences:
            board, pieces = exp.state
            next_board, next_pieces = exp.next_state
            
            board_t, pieces_t = self.encode_state(board, pieces)
            next_board_t, next_pieces_t = self.encode_state(next_board, next_pieces)
            
            batch_states.append(board_t)
            batch_pieces.append(pieces_t)
            batch_rewards.append(exp.reward)
            batch_next_states.append(next_board_t)
            batch_next_pieces.append(next_pieces_t)
            batch_dones.append(exp.done)
            
        # Stack batches
        states_board = torch.stack(batch_states)
        states_pieces = torch.stack(batch_pieces)
        rewards = torch.FloatTensor(batch_rewards).to(self.device)
        next_states_board = torch.stack(batch_next_states)
        next_states_pieces = torch.stack(batch_next_pieces)
        dones = torch.FloatTensor(batch_dones).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states_board, states_pieces).squeeze()
        
        # Next Q values (Double DQN)
        with torch.no_grad():
            next_q_target = self.target_net(next_states_board, next_states_pieces).squeeze()
            targets = rewards + self.gamma * next_q_target * (1 - dones)
            
        # TD errors for prioritization
        td_errors = torch.abs(current_q - targets).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Weighted loss
        loss = (weights_tensor * F.mse_loss(current_q, targets, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self):
        """Update target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def save(self, path):
        """Save model."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        
    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']