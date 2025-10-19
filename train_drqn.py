#!/usr/bin/env python3
"""
DRQN Trading Training Script

A command-line interface for training the DRQN model on stock data.
Based on the published approach from: https://github.com/conditionWang/DRQN_Stock_Trading
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import math
from torch.utils.tensorboard import SummaryWriter
import yfinance as yf
from datetime import datetime, timedelta
import os
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DRQN(nn.Module):
    """
    Deep Recurrent Q-Network for trading.

    Architecture following the published paper:
    - Layer 1: Linear(state_size, 256) + ELU
    - Layer 2: Linear(256, 256) + ELU
    - Layer 3: LSTM(256, 256)
    - Layer 4: Linear(256, 3) for 3 actions (bear, hold, bull)
    """

    def __init__(self, state_size=14, action_size=3):
        super(DRQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # First two layers
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)

        # LSTM layer
        self.lstm = nn.LSTM(256, 256, batch_first=True)

        # Output layer
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, state_size)
        batch_size, seq_len, _ = x.shape

        # Reshape for processing
        x = x.view(-1, self.state_size)  # (batch_size * seq_len, state_size)

        # First two layers
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))

        # Reshape back for LSTM
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 256)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Take the last output
        x = lstm_out[:, -1, :]  # (batch_size, 256)

        # Output layer
        x = self.fc3(x)  # (batch_size, 3)

        return x

class TradingEnvironment:
    """
    Trading environment following the published DRQN approach.

    Actions:
    - 0: Bear (-1) - Short position
    - 1: Hold (0) - No position
    - 2: Bull (1) - Long position
    """

    def __init__(self, data, initial_balance=100000, trade_size=10000, spread=0.005, max_position_ratio=0.3):
        self.data = data
        self.initial_balance = initial_balance
        self.trade_size = trade_size
        self.spread = spread
        self.max_position_ratio = max_position_ratio  # Max 30% of portfolio in one position

        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.balance = self.initial_balance
        self.position = 0  # Current position: -1, 0, or 1
        self.current_step = 0
        self.portfolio_values = [self.initial_balance]

        return self._get_state()

    def _get_state(self):
        """Get current state features following the paper's approach."""
        if self.current_step >= len(self.data) - 1:
            return None

        # Get current price
        current_price = float(self.data.iloc[self.current_step]['Close'])

        # Create state features (simplified version of the paper's approach)
        # The paper uses 8 delayed log returns + time features + previous action
        price_features = []

        # Add price-based features
        if self.current_step >= 8:
            # 8 delayed log returns (as in the paper)
            for i in range(8):
                if self.current_step - i - 1 >= 0:
                    prev_price = float(self.data.iloc[self.current_step - i - 1]['Close'])
                    log_return = np.log(current_price / prev_price)
                    price_features.append(log_return)
                else:
                    price_features.append(0.0)
        else:
            # Pad with zeros if not enough history
            price_features.extend([0.0] * 8)

        # Add time features (simplified)
        price_features.extend([
            float(self.current_step / len(self.data)),  # Normalized time
            float(self.position),  # Previous action
            float(self.balance / self.initial_balance),  # Normalized balance
        ])

        # Pad to 14 features total
        while len(price_features) < 14:
            price_features.append(0.0)

        return np.array(price_features[:14], dtype=np.float32)

    def step(self, action):
        """Execute action following the published paper's approach."""
        if self.current_step >= len(self.data) - 1:
            return None, 0, True

        current_price = float(self.data.iloc[self.current_step]['Close'])
        next_price = float(self.data.iloc[self.current_step + 1]['Close'])

        # Convert action to position (-1, 0, 1) following paper's logic
        new_position = action - 1  # 0->-1, 1->0, 2->1

        # Calculate price change
        price_change = (next_price - current_price) / current_price

        # Paper's trading logic:
        # - Bear (-1): Borrow stocks, sell at open, buy back at close
        # - Bull (1): Borrow money, buy at open, sell at close
        # - Hold (0): Do nothing

        # Calculate profit/loss from the action
        if new_position == 1:  # Bull transaction
            # Borrow money, buy at current_price, sell at next_price
            profit_loss = (next_price - current_price) * self.trade_size
        elif new_position == -1:  # Bear transaction
            # Borrow stocks, sell at current_price, buy back at next_price
            profit_loss = (current_price - next_price) * self.trade_size
        else:  # Hold (0)
            profit_loss = 0

        # Commission cost (only for non-hold actions)
        commission_cost = 0
        if new_position != 0:
            commission_cost = self.trade_size * self.spread

        # Net profit/loss
        net_pnl = profit_loss - commission_cost

        # Update balance (paper's approach: balance absorbs profits/losses)
        self.balance += net_pnl

        # Update position
        self.position = new_position

        # Update portfolio value
        portfolio_value = self.balance
        self.portfolio_values.append(portfolio_value)

        # Reward is the actual portfolio change (paper's approach)
        reward = net_pnl

        self.current_step += 1

        # Check if done
        done = self.current_step >= len(self.data) - 1

        next_state = self._get_state() if not done else None

        return next_state, reward, done

class DRQNAgent:
    """
    DRQN Agent with experience replay following the published approach.
    """

    def __init__(self, state_size=14, action_size=3, lr=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.9995, batch_size=32, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Networks
        self.q_network = DRQN(state_size, action_size).to(device)
        self.target_network = DRQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Experience replay
        self.memory = deque(maxlen=memory_size)

        # Action tracking for balanced exploration
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.total_actions = 0

    def act(self, state, training=True):
        """Choose action using improved epsilon-greedy policy with balanced exploration."""
        if training and np.random.random() <= self.epsilon:
            # Balanced exploration - favor under-explored actions
            action_probs = [1.0 / (self.action_counts[i] + 1) for i in range(self.action_size)]
            total_prob = sum(action_probs)
            action_probs = [p / total_prob for p in action_probs]
            action = np.random.choice(self.action_size, p=action_probs)
        else:
            with torch.no_grad():
                # Convert single state to sequence format (batch_size=1, seq_len=1, state_size)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()

        # Track action for balanced exploration
        if training:
            self.action_counts[action] += 1
            self.total_actions += 1

        return action

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def remember_action_augmentation(self, state, all_actions, all_rewards, next_state, done):
        """Store Action Augmentation experiences (paper's key innovation)."""
        # Store transitions for ALL possible actions, not just the chosen one
        for action, reward in zip(all_actions, all_rewards):
            self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Filter out None next_states
        valid_indices = [i for i, ns in enumerate(next_states) if ns is not None]
        if len(valid_indices) < self.batch_size // 2:
            return 0.0

        # Use only valid transitions
        states = [states[i] for i in valid_indices]
        actions = [actions[i] for i in valid_indices]
        rewards = [rewards[i] for i in valid_indices]
        next_states = [next_states[i] for i in valid_indices]
        dones = [dones[i] for i in valid_indices]

        # Convert to tensors with proper sequence format
        states = torch.FloatTensor(states).unsqueeze(1).to(device)  # (batch, seq_len=1, state_size)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(device)  # (batch, seq_len=1, state_size)
        dones = torch.BoolTensor(dones).to(device)

        # Get current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Get next Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

def load_data(ticker, start_date, end_date):
    """Load stock data using yfinance."""
    print(f"Loading data for {ticker} from {start_date} to {end_date}")

    data = yf.download(ticker, start=start_date, end=end_date)

    # Handle MultiIndex columns properly
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    print(f"Downloaded {len(data)} days of data")
    return data

def train_model(args):
    """Train the DRQN model."""
    print(f"Starting DRQN training with parameters:")
    print(f"  Ticker: {args.ticker}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Initial Balance: ${args.initial_balance:,.0f}")
    print(f"  Trade Size: ${args.trade_size:,.0f}")
    print(f"  Spread: {args.spread}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Device: {device}")

    # Load data
    data = load_data(args.ticker, args.start_date, args.end_date)

    # Initialize environment and agent
    env = TradingEnvironment(
        data,
        initial_balance=args.initial_balance,
        trade_size=args.trade_size,
        spread=args.spread
    )

    agent = DRQNAgent(
        lr=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        memory_size=args.memory_size
    )

    # Setup logging with unique run name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"drqn_{args.ticker}_ep{args.episodes}_lr{args.learning_rate}_bal{args.initial_balance}_{timestamp}"

    # Create separate directories for TensorBoard logs and saved files
    tensorboard_dir = f"{args.log_dir}/tensorboard"
    models_dir = f"{args.log_dir}/models"
    results_dir = f"{args.log_dir}/results"

    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    writer = SummaryWriter(f"{tensorboard_dir}/{run_name}")

    # Training loop with early stopping
    episode_rewards = []
    episode_portfolios = []
    episode_losses = []
    best_portfolio = -float('inf')
    patience_counter = 0
    early_stop_patience = 50  # Stop if no improvement for 50 episodes

    print("\nStarting training...")
    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0

        while state is not None:
            # Action Augmentation: Evaluate ALL possible actions (paper's key innovation)
            all_actions = [0, 1, 2]  # Bear, Hold, Bull
            all_rewards = []

            # Calculate rewards for all possible actions
            for action in all_actions:
                # Temporarily execute each action to get reward
                temp_env = env
                temp_state = temp_env.current_step
                temp_balance = temp_env.balance
                temp_position = temp_env.position

                # Simulate action
                if temp_env.current_step >= len(temp_env.data) - 1:
                    reward = 0
                else:
                    current_price = float(temp_env.data.iloc[temp_env.current_step]['Close'])
                    next_price = float(temp_env.data.iloc[temp_env.current_step + 1]['Close'])

                    # Calculate profit/loss for this action
                    new_position = action - 1  # 0->-1, 1->0, 2->1

                    if new_position == 1:  # Bull
                        profit_loss = (next_price - current_price) * temp_env.trade_size
                    elif new_position == -1:  # Bear
                        profit_loss = (current_price - next_price) * temp_env.trade_size
                    else:  # Hold
                        profit_loss = 0

                    commission_cost = 0
                    if new_position != 0:
                        commission_cost = temp_env.trade_size * temp_env.spread

                    reward = profit_loss - commission_cost

                all_rewards.append(reward)

            # Agent selects action (can use epsilon-greedy or other strategy)
            action = agent.act(state)
            reward = all_rewards[action]

            # Environment executes the chosen action
            next_state, actual_reward, done = env.step(action)

            # Store Action Augmentation experiences
            if next_state is not None:
                agent.remember_action_augmentation(state, all_actions, all_rewards, next_state, done)

            total_reward += reward
            state = next_state
            step_count += 1

            if done:
                break

        # Train the agent on collected experiences
        loss = agent.replay()

        # Update target network periodically
        if episode % args.update_target_freq == 0:
            agent.update_target_network()

        # Log results
        final_portfolio = env.portfolio_values[-1]
        episode_rewards.append(total_reward)
        episode_portfolios.append(final_portfolio)
        episode_losses.append(loss if loss else 0)

        # TensorBoard logging
        writer.add_scalar('Reward/Episode', total_reward, episode)
        writer.add_scalar('Portfolio/Value', final_portfolio, episode)
        writer.add_scalar('Loss/Training', loss if loss else 0, episode)
        writer.add_scalar('Epsilon/Value', agent.epsilon, episode)

        # Early stopping check
        if final_portfolio > best_portfolio:
            best_portfolio = final_portfolio
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at episode {episode} (no improvement for {early_stop_patience} episodes)")
            break

        # Print progress with action distribution
        if episode % args.print_freq == 0:
            action_dist = {i: agent.action_counts[i] / max(agent.total_actions, 1) * 100 for i in range(3)}
            print(f"Episode {episode:4d} | Reward: {total_reward:8.2f} | Portfolio: ${final_portfolio:10,.2f} | Epsilon: {agent.epsilon:.3f} | Actions: Bear:{action_dist[0]:.1f}% Hold:{action_dist[1]:.1f}% Bull:{action_dist[2]:.1f}%")

    writer.close()

    # Save results
    results = {
        'episode_rewards': episode_rewards,
        'episode_portfolios': episode_portfolios,
        'episode_losses': episode_losses,
        'final_portfolio': episode_portfolios[-1],
        'total_return': (episode_portfolios[-1] - args.initial_balance) / args.initial_balance * 100
    }

    # Save model
    if args.save_model:
        model_path = f"{models_dir}/{run_name}_model.pt"
        torch.save(agent.q_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Save results
    results_path = f"{results_dir}/{run_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Plot results
    if args.plot:
        plot_path = f"{results_dir}/{run_name}_plot.png"
        plot_results(results, args.ticker, plot_path)

    print(f"\nTraining completed!")
    print(f"Final Portfolio: ${episode_portfolios[-1]:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Results saved to {results_path}")

    return results

def plot_results(results, ticker, plot_path):
    """Plot training results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Episode rewards
    axes[0, 0].plot(results['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)

    # Portfolio values
    axes[0, 1].plot(results['episode_portfolios'])
    axes[0, 1].set_title('Portfolio Values')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].grid(True)

    # Training loss
    axes[1, 0].plot(results['episode_losses'])
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)

    # Return percentage
    returns = [(p - results['episode_portfolios'][0]) / results['episode_portfolios'][0] * 100
               for p in results['episode_portfolios']]
    axes[1, 1].plot(returns)
    axes[1, 1].set_title('Portfolio Return (%)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Return (%)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Train DRQN model for stock trading')

    # Data parameters
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD), defaults to today')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon for exploration')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum epsilon')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--memory_size', type=int, default=10000, help='Replay memory size')
    parser.add_argument('--update_target_freq', type=int, default=10, help='Target network update frequency')

    # Environment parameters
    parser.add_argument('--initial_balance', type=float, default=100000, help='Initial portfolio balance')
    parser.add_argument('--trade_size', type=float, default=10000, help='Trade size per transaction')
    parser.add_argument('--spread', type=float, default=0.005, help='Commission spread')

    # Output parameters
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logs and outputs')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    parser.add_argument('--plot', action='store_true', help='Generate plots')

    args = parser.parse_args()

    # Set end date to today if not provided
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')

    # Train model
    results = train_model(args)

if __name__ == '__main__':
    main()
