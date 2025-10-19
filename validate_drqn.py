#!/usr/bin/env python3
"""
DRQN Model Validation Script

Evaluate trained DRQN models on separate datasets for validation.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
import math
import json
import yfinance as yf
from datetime import datetime, timedelta
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DRQN(nn.Module):
    """
    Deep Recurrent Q-Network for trading.
    Same architecture as training script.
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
    Trading environment for validation (same as training).
    """

    def __init__(self, data, initial_balance=100000, trade_size=10000, spread=0.005):
        self.data = data
        self.initial_balance = initial_balance
        self.trade_size = trade_size
        self.spread = spread

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
    DRQN Agent for validation (no training, just inference).
    """

    def __init__(self, state_size=14, action_size=3):
        self.state_size = state_size
        self.action_size = action_size

        # Load the trained model
        self.q_network = DRQN(state_size, action_size).to(device)

    def load_model(self, model_path):
        """Load trained model weights."""
        self.q_network.load_state_dict(torch.load(model_path, map_location=device))
        self.q_network.eval()  # Set to evaluation mode
        print(f"Model loaded from {model_path}")

    def act(self, state):
        """Choose action using greedy policy (no exploration)."""
        with torch.no_grad():
            # Convert single state to sequence format (batch_size=1, seq_len=1, state_size)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

def load_data(ticker, start_date, end_date):
    """Load stock data using yfinance."""
    print(f"Loading validation data for {ticker} from {start_date} to {end_date}")

    data = yf.download(ticker, start=start_date, end=end_date)

    # Handle MultiIndex columns properly
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    print(f"Downloaded {len(data)} days of data")
    return data

def validate_model(args):
    """Validate the trained DRQN model."""
    print(f"Starting DRQN validation:")
    print(f"  Model: {args.model_path}")
    print(f"  Ticker: {args.ticker}")
    print(f"  Validation Period: {args.start_date} to {args.end_date}")
    print(f"  Initial Balance: ${args.initial_balance:,.0f}")
    print(f"  Trade Size: ${args.trade_size:,.0f}")
    print(f"  Spread: {args.spread}")
    print(f"  Device: {device}")

    # Load validation data
    data = load_data(args.ticker, args.start_date, args.end_date)

    # Initialize environment and agent
    env = TradingEnvironment(
        data,
        initial_balance=args.initial_balance,
        trade_size=args.trade_size,
        spread=args.spread
    )

    agent = DRQNAgent()
    agent.load_model(args.model_path)

    # Run validation
    print("\nRunning validation...")
    state = env.reset()
    total_reward = 0
    actions_taken = []
    portfolio_values = []
    step_count = 0

    while state is not None:
        # Agent selects action (greedy, no exploration)
        action = agent.act(state)
        actions_taken.append(action)

        # Environment executes action
        next_state, reward, done = env.step(action)

        total_reward += reward
        portfolio_values.append(env.balance)
        state = next_state
        step_count += 1

        if done:
            break

    # Calculate metrics
    final_portfolio = env.portfolio_values[-1]
    total_return = (final_portfolio - args.initial_balance) / args.initial_balance * 100

    # Calculate buy-and-hold baseline
    initial_price = float(data.iloc[0]['Close'])
    final_price = float(data.iloc[-1]['Close'])
    buy_hold_return = (final_price - initial_price) / initial_price * 100

    # Calculate action distribution
    action_counts = {0: 0, 1: 0, 2: 0}
    for action in actions_taken:
        action_counts[action] += 1

    action_distribution = {
        'bear': action_counts[0] / len(actions_taken) * 100,
        'hold': action_counts[1] / len(actions_taken) * 100,
        'bull': action_counts[2] / len(actions_taken) * 100
    }

    # Create results
    results = {
        'ticker': args.ticker,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'initial_balance': args.initial_balance,
        'final_portfolio': final_portfolio,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'outperformance': total_return - buy_hold_return,
        'total_reward': total_reward,
        'steps': step_count,
        'action_distribution': action_distribution,
        'portfolio_values': portfolio_values,
        'actions_taken': actions_taken
    }

    # Print results
    print(f"\nValidation Results:")
    print(f"  Final Portfolio: ${final_portfolio:,.2f}")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"  Outperformance: {total_return - buy_hold_return:.2f}%")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Steps: {step_count}")
    print(f"  Action Distribution:")
    print(f"    Bear: {action_distribution['bear']:.1f}%")
    print(f"    Hold: {action_distribution['hold']:.1f}%")
    print(f"    Bull: {action_distribution['bull']:.1f}%")

    # Save results
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = f"{args.output_dir}/validation_{args.ticker}_{args.start_date}_{args.end_date}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

    # Generate plots
    if args.plot:
        plot_validation_results(results, args.ticker, args.output_dir)

    return results

def plot_validation_results(results, ticker, output_dir):
    """Plot validation results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Portfolio progression
    axes[0, 0].plot(results['portfolio_values'])
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].grid(True)

    # Action distribution
    actions = ['Bear', 'Hold', 'Bull']
    counts = [results['action_distribution']['bear'],
              results['action_distribution']['hold'],
              results['action_distribution']['bull']]
    axes[0, 1].bar(actions, counts)
    axes[0, 1].set_title('Action Distribution')
    axes[0, 1].set_ylabel('Percentage (%)')
    axes[0, 1].grid(True)

    # Returns comparison
    returns = ['DRQN', 'Buy & Hold']
    values = [results['total_return'], results['buy_hold_return']]
    colors = ['blue', 'orange']
    axes[1, 0].bar(returns, values, color=colors)
    axes[1, 0].set_title('Return Comparison')
    axes[1, 0].set_ylabel('Return (%)')
    axes[1, 0].grid(True)

    # Action sequence (first 100 steps)
    action_sequence = results['actions_taken'][:100]
    axes[1, 1].plot(action_sequence, marker='o', markersize=2)
    axes[1, 1].set_title('Action Sequence (First 100 Steps)')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Action (0=Bear, 1=Hold, 2=Bull)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plot_path = f"{output_dir}/validation_{ticker}_{results['start_date']}_{results['end_date']}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Validate trained DRQN model')

    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pt file)')

    # Data parameters
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD), defaults to today')

    # Environment parameters
    parser.add_argument('--initial_balance', type=float, default=100000, help='Initial portfolio balance')
    parser.add_argument('--trade_size', type=float, default=10000, help='Trade size per transaction')
    parser.add_argument('--spread', type=float, default=0.005, help='Commission spread')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='validation_results', help='Directory for validation outputs')
    parser.add_argument('--save_results', action='store_true', help='Save validation results')
    parser.add_argument('--plot', action='store_true', help='Generate validation plots')

    args = parser.parse_args()

    # Set end date to today if not provided
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')

    # Validate model
    results = validate_model(args)

if __name__ == '__main__':
    main()
