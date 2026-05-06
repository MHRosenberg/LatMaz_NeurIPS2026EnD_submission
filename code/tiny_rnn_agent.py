"""
Tiny RNN agent for latent maze navigation.

Inspired by Ji-An, Benna & Mattar (Nature 2025): "Discovering cognitive
strategies with tiny recurrent neural networks."

Core idea: very small GRU networks (1-4 hidden units) can capture the
cognitive algorithms underlying individual decision-making. We adapt
this to maze navigation:

  Input:  prev_ego_action (one-hot 4) + prev_ego_obs (binary 4) + prev_reward (1) = 9 dims
  Hidden: GRU with d units (d = 1, 2, or 4)
  Output: P(ego_action) via softmax over 4 ego actions (F/L/R/B)

Two training modes:
  A) Mouse-fit: maximize log-likelihood of observed mouse actions
  B) Meta-RL: maximize cumulative reward via REINFORCE across mazes

References:
  Ji-An, L., Benna, M. K., & Mattar, M. G. (2025). Discovering cognitive
  strategies with tiny recurrent neural networks. Nature, 644, 993-1001.
"""

import os
import sys
import ast
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_this_dir = os.path.dirname(os.path.abspath(__file__))
_code_in_mr = os.path.join(_this_dir, '..', '..', 'code')
if _code_in_mr not in sys.path:
    sys.path.insert(0, _code_in_mr)

from utils_latMaz import (
    get_adj_states,
    displacement_to_compass_heading,
    allo_actions_one_hot_dict,
)
from advanced_agents import EGO_TO_ALLO, ALLO_TO_EGO
from intermediate_agents import _EgoSimulator


# =============================================================================
# TinyGRU Network
# =============================================================================

class TinyGRU(nn.Module):
    """Tiny GRU network for action prediction.

    Architecture matches Ji-An et al. (2025):
      input (9-dim) -> GRU (d units) -> linear -> softmax (4 actions)

    For d=1, uses switching GRU variant (input-dependent weights) as in
    the paper, since vanilla GRU with 1 unit has limited expressivity.
    """

    def __init__(self, hidden_dim=2, n_actions=4, input_dim=9,
                 use_switching=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.input_dim = input_dim
        self.use_switching = use_switching and hidden_dim == 1

        if self.use_switching:
            # Switching GRU: separate weights per discrete input pattern.
            # We discretize the 9-dim input into (action, obs, reward) combos.
            # 4 actions × 16 obs × 2 rewards = 128 possible inputs.
            n_input_combos = 128
            self.gru_weights = nn.ModuleList([
                nn.GRUCell(1, hidden_dim) for _ in range(n_input_combos)
            ])
            self.input_proj = None
        else:
            self.gru = nn.GRUCell(input_dim, hidden_dim)

        self.readout = nn.Linear(hidden_dim, n_actions)

    def _input_to_idx(self, x):
        """Convert 9-dim input to discrete index for switching GRU."""
        # x: (batch, 9) = [action_onehot(4), obs(4), reward(1)]
        action = x[:, :4].argmax(dim=1)    # 0-3
        obs = (x[:, 4:8] * torch.tensor([1, 2, 4, 8],
               device=x.device)).sum(dim=1).long()  # 0-15
        reward = x[:, 8].long()             # 0-1
        return action * 32 + obs * 2 + reward  # 0-127

    def forward(self, x, h):
        """One step of the GRU.

        Args:
            x: (batch, input_dim) input vector
            h: (batch, hidden_dim) previous hidden state

        Returns:
            logits: (batch, n_actions) action logits
            h_new: (batch, hidden_dim) new hidden state
        """
        if self.use_switching:
            # Switching GRU: select weights based on input
            idx = self._input_to_idx(x)
            h_new = torch.zeros_like(h)
            for i in range(x.shape[0]):
                dummy_input = h[i:i+1, :]  # reuse h as dummy
                h_new[i:i+1] = self.gru_weights[idx[i]](
                    torch.zeros(1, 1, device=x.device), h[i:i+1])
            h = h_new
        else:
            h = self.gru(x, h)

        logits = self.readout(h)
        return logits, h

    def init_hidden(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_dim, device=device)


# =============================================================================
# Mouse Data Loader
# =============================================================================

def load_mouse_sequences(yoking_csv, raw_data_dir=None):
    """Load mouse action/observation/reward sequences from raw CSVs.

    Returns list of dicts, each containing:
      - 'animal_id': str
      - 'exp_moment': str
      - 'actions': list of ego action indices (0=F, 1=B, 2=L, 3=R)
      - 'observations': list of 4-bit ego obs tuples
      - 'rewards': list of float rewards
      - 'adj_file': str
      - 'st_pos_file': str
      - 'start_node': int
      - 'n_actions': int
    """
    import pandas as pd
    import re

    yoking_df = pd.read_csv(yoking_csv)
    sequences = []

    for _, row in yoking_df.iterrows():
        csv_path = row.get('csv_data_path', '')
        if not csv_path or not os.path.exists(csv_path):
            continue

        try:
            raw_df = pd.read_csv(csv_path)
        except Exception:
            continue

        # Extract exp_moment from path
        match = re.search(r'(\d{6}-\d{6})', str(csv_path))
        exp_mom = match.group(1) if match else ''

        # Parse states_visited from yoking row
        states = row.get('states_visited', '[]')
        if isinstance(states, str):
            states = ast.literal_eval(states)

        # Need at least 2 states for 1 action
        if len(states) < 2:
            continue

        # Extract ego actions from raw CSV
        if 'action' in raw_df.columns:
            action_map = {'forwards': 0, 'backwards': 1, 'left': 2, 'right': 3,
                          'forward': 0, 'backward': 1}
            actions = []
            for a in raw_df['action'].values:
                a_str = str(a).lower().strip()
                if a_str in action_map:
                    actions.append(action_map[a_str])
                else:
                    actions.append(0)  # default forward
        else:
            continue

        # Extract rewards (binary: did reward count increase?)
        if 'n_rewards' in raw_df.columns:
            n_rwd = raw_df['n_rewards'].values.astype(float)
            rewards = [0.0]  # first step has no prior reward
            for t in range(1, len(n_rwd)):
                rewards.append(1.0 if n_rwd[t] > n_rwd[t-1] else 0.0)
        else:
            rewards = [0.0] * len(actions)

        # Extract ego observations from lights_on
        observations = []
        if 'lights_on' in raw_df.columns:
            for val in raw_df['lights_on'].values:
                if isinstance(val, str):
                    # Parse [True, False, True, True] etc
                    bits = [1 if 'true' in s.lower() else 0
                            for s in val.strip('[]').split(',')]
                    # lights_on is [N,S,E,W] = walls present
                    # Invert: passable = not wall
                    obs = tuple(1 - b for b in bits[:4])
                else:
                    obs = (1, 1, 1, 1)
                observations.append(obs)
        else:
            # Reconstruct from adjacency matrix
            observations = [(1, 1, 1, 1)] * len(actions)

        # Trim to consistent length
        n = min(len(actions), len(rewards), len(observations))
        sequences.append({
            'animal_id': str(row.get('animal_ID', '')),
            'exp_moment': exp_mom,
            'actions': actions[:n],
            'observations': observations[:n],
            'rewards': rewards[:n],
            'adj_file': str(row.get('adj_file', '')),
            'st_pos_file': str(row.get('st_pos_file', '')),
            'start_node': int(row.get('start_state', 0)),
            'n_actions': n,
        })

    return sequences


def sequences_to_tensors(sequences, device='cpu'):
    """Convert mouse sequences to padded tensors for batch training.

    Returns:
        inputs: (n_seqs, max_len, 9) — input at each step
        targets: (n_seqs, max_len) — target action index
        lengths: (n_seqs,) — sequence lengths
    """
    max_len = max(s['n_actions'] for s in sequences)
    n_seqs = len(sequences)

    inputs = torch.zeros(n_seqs, max_len, 9, device=device)
    targets = torch.full((n_seqs, max_len), -1, dtype=torch.long, device=device)
    lengths = torch.zeros(n_seqs, dtype=torch.long, device=device)

    for i, seq in enumerate(sequences):
        n = seq['n_actions']
        lengths[i] = n

        for t in range(n):
            targets[i, t] = seq['actions'][t]

            if t == 0:
                # First step: no previous action/obs/reward
                # Use zeros (neutral initialization)
                continue

            # Previous action (one-hot)
            prev_action = seq['actions'][t - 1]
            inputs[i, t, prev_action] = 1.0

            # Previous observation (4-bit binary)
            prev_obs = seq['observations'][t - 1]
            for j in range(4):
                inputs[i, t, 4 + j] = float(prev_obs[j])

            # Previous reward
            inputs[i, t, 8] = seq['rewards'][t - 1]

    return inputs, targets, lengths


# =============================================================================
# Training
# =============================================================================

def train_tiny_rnn(model, sequences, n_epochs=200, lr=0.005,
                   l1_coef=1e-3, device='cpu', verbose=True):
    """Train TinyGRU on mouse behavioral sequences.

    Args:
        model: TinyGRU network
        sequences: list of sequence dicts from load_mouse_sequences
        n_epochs: number of training epochs
        lr: learning rate
        l1_coef: L1 regularization on recurrent weights
        device: torch device
        verbose: print progress

    Returns:
        losses: list of per-epoch losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    inputs, targets, lengths = sequences_to_tensors(sequences, device)
    n_seqs = len(sequences)

    losses = []
    best_loss = float('inf')
    patience_counter = 0
    PATIENCE = 50

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        h = model.init_hidden(n_seqs, device)

        epoch_loss = torch.tensor(0.0, device=device)
        n_valid = 0

        max_len = inputs.shape[1]
        for t in range(max_len):
            x = inputs[:, t, :]  # (n_seqs, 9)
            target = targets[:, t]  # (n_seqs,)

            logits, h = model(x, h)
            h = h.detach()  # Truncated BPTT per step (following paper)

            # Mask: only compute loss for valid timesteps
            mask = (target >= 0)
            if mask.sum() == 0:
                continue

            step_loss = F.cross_entropy(logits[mask], target[mask],
                                        reduction='sum')
            epoch_loss = epoch_loss + step_loss
            n_valid += mask.sum().item()

        avg_loss = epoch_loss.item() / max(n_valid, 1)
        epoch_loss = epoch_loss / max(n_valid, 1)

        # L1 regularization on recurrent weights
        l1_reg = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if 'weight_hh' in name:
                l1_reg = l1_reg + param.abs().sum()

        (epoch_loss + l1_coef * l1_reg).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(avg_loss)

        # Early stopping
        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            if verbose:
                print(f"  Early stop at epoch {epoch+1}, loss={avg_loss:.4f}")
            break

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    return losses


# =============================================================================
# TinyRNN Agent (compatible with run_episode interface)
# =============================================================================

class TinyRNNAgent:
    """Tiny RNN agent for maze navigation.

    Uses a trained TinyGRU to select actions in an egocentric maze.
    Compatible with the planning-agent run_episode() interface.

    Args:
        hidden_dim: number of GRU hidden units (1, 2, or 4)
        prevent_backward: if True, mask Backward action when other dirs available
        temperature: softmax temperature for action sampling (1.0 = trained dist)
        weights_path: path to saved model weights (optional)
        model: pre-trained TinyGRU model (optional, overrides weights_path)
    """

    def __init__(self, hidden_dim=2, prevent_backward=True,
                 temperature=1.0, weights_path=None, model=None,
                 n_think_steps=0):
        self.hidden_dim = hidden_dim
        self.prevent_backward = prevent_backward
        self.temperature = temperature
        self.n_think_steps = n_think_steps  # DRC-style extra internal steps

        if model is not None:
            self.model = model
        elif weights_path is not None:
            self.model = TinyGRU(hidden_dim=hidden_dim)
            self.model.load_state_dict(torch.load(weights_path,
                                                   weights_only=True))
        else:
            self.model = TinyGRU(hidden_dim=hidden_dim)

        self.model.eval()

    def run_episode(
        self,
        adj_mat: np.ndarray,
        node_positions: np.ndarray,
        start_node: int,
        rewarded_nodes: np.ndarray,
        n_actions: int,
        min_rewarded_states: int = 2,
        seed: int = 42,
        verbose: bool = False,
        prevent_reverse: bool = False,
        return_trajectory: bool = False,
    ) -> float:
        """Run episode using the trained TinyGRU policy."""
        np.random.seed(seed)
        torch.manual_seed(seed)

        sim = _EgoSimulator(
            adj_mat, node_positions, start_node, rewarded_nodes,
            min_rewarded_states=min_rewarded_states,
            prevent_reverse=prevent_reverse,
        )
        obs = sim.reset()
        trajectory = [] if return_trajectory else None

        h = self.model.init_hidden(1)
        prev_action = 0  # neutral: Forward
        prev_reward = 0.0

        for step in range(n_actions):
            # Build input: prev_action_onehot(4) + prev_obs(4) + prev_reward(1)
            x = torch.zeros(1, 9)
            if step > 0:
                x[0, prev_action] = 1.0
                for j in range(4):
                    x[0, 4 + j] = float(obs[j])
                x[0, 8] = prev_reward

            with torch.no_grad():
                logits, h = self.model(x, h)
                # DRC-style think steps: re-feed same input N extra times
                for _ in range(self.n_think_steps):
                    logits, h = self.model(x, h)

            # Apply temperature
            logits = logits / self.temperature

            # Get valid ego actions from current observation
            valid_mask = torch.zeros(4)
            for a in range(4):
                if obs[a]:  # passable direction
                    valid_mask[a] = 1.0

            # Optionally mask backward (action 1)
            if self.prevent_backward and valid_mask.sum() > 1 and valid_mask[1] > 0:
                valid_mask[1] = 0.0

            # Mask invalid actions
            logits = logits.squeeze(0)
            logits[valid_mask == 0] = float('-inf')

            # Sample action
            probs = F.softmax(logits, dim=0)
            action = torch.multinomial(probs, 1).item()

            # Execute
            new_obs, reward, moved = sim.step(action)

            if return_trajectory:
                trajectory.append((sim.real_node, reward, 'tiny_rnn'))

            prev_action = action
            prev_reward = reward
            obs = new_obs

        if return_trajectory:
            return sim.total_reward, trajectory
        return sim.total_reward


# =============================================================================
# Meta-RL Training (Mode B)
# =============================================================================

def train_meta_rl(model, adj_mats, node_positions_list, start_nodes,
                  rewarded_nodes_list, n_actions_list,
                  min_rewarded_states_list,
                  n_epochs=100, lr=1e-3, gamma=0.99,
                  prevent_backward=True, device='cpu', verbose=True):
    """Train TinyGRU via REINFORCE across multiple maze episodes.

    This is Mode B: find the best policy a tiny RNN can learn for
    maze navigation (rather than imitating the mouse).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_mazes = len(adj_mats)

    epoch_rewards = []
    for epoch in range(n_epochs):
        total_reward = 0.0
        total_loss = torch.tensor(0.0, device=device)
        n_episodes = 0

        # Shuffle maze order each epoch
        order = list(range(n_mazes))
        random.shuffle(order)

        for idx in order:
            sim = _EgoSimulator(
                adj_mats[idx], node_positions_list[idx],
                start_nodes[idx], rewarded_nodes_list[idx],
                min_rewarded_states=min_rewarded_states_list[idx],
                prevent_reverse=False,
            )
            obs = sim.reset()
            h = model.init_hidden(1, device)

            log_probs = []
            rewards_list = []
            prev_action = 0
            prev_reward = 0.0

            for step in range(n_actions_list[idx]):
                x = torch.zeros(1, 9, device=device)
                if step > 0:
                    x[0, prev_action] = 1.0
                    for j in range(4):
                        x[0, 4 + j] = float(obs[j])
                    x[0, 8] = prev_reward

                logits, h = model(x, h)
                logits = logits.squeeze(0)

                # Mask invalid actions
                valid_mask = torch.zeros(4, device=device)
                for a in range(4):
                    if obs[a]:
                        valid_mask[a] = 1.0
                if prevent_backward and valid_mask.sum() > 1 and valid_mask[1] > 0:
                    valid_mask[1] = 0.0
                logits[valid_mask == 0] = float('-inf')

                probs = F.softmax(logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))

                new_obs, reward, moved = sim.step(action.item())
                rewards_list.append(reward)

                prev_action = action.item()
                prev_reward = reward
                obs = new_obs

            # Compute returns
            returns = []
            R = 0.0
            for r in reversed(rewards_list):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, device=device)
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # REINFORCE loss
            ep_loss = torch.tensor(0.0, device=device)
            for lp, ret in zip(log_probs, returns):
                ep_loss = ep_loss - lp * ret

            total_loss = total_loss + ep_loss
            total_reward += sim.total_reward
            n_episodes += 1

        optimizer.zero_grad()
        (total_loss / n_episodes).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        avg_rwd = total_reward / n_episodes
        epoch_rewards.append(avg_rwd)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: avg_reward={avg_rwd:.3f}")

    return epoch_rewards
