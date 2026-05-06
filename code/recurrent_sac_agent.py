"""
Recurrent SAC-Discrete agent for latent maze navigation.

Based on Ni et al. (2022): "Recurrent Model-Free RL Can Be a Strong Baseline
for Many POMDPs" — separate actor/critic GRU encoders with off-policy SAC
backbone.

SAC-Discrete formulation from Christodoulou (2019): "Soft Actor-Critic for
Discrete Action Settings."

Architecture:
  Input:  prev_ego_action (one-hot 4) + prev_ego_obs (binary 4) + prev_reward (1) = 9 dims
  Actor:  GRU(9 → D) → Linear(D → 4) → softmax → π(a|s)
  Critic1: GRU(9 → D) → Linear(D → 4) → Q1(s, a)
  Critic2: GRU(9 → D) → Linear(D → 4) → Q2(s, a)

All three GRUs have separate parameters (Ni et al. key finding).
Meta-RL training: SAC-Discrete across all 61 yoked sessions.
"""

import os
import sys
import random
from collections import deque

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
# RecurrentSACNetwork
# =============================================================================

class RecurrentSACNetwork(nn.Module):
    """Recurrent SAC-Discrete with separate actor and twin-critic GRUs.

    Architecture (Ni et al. 2022):
      Actor:   GRU(input_dim → hidden_dim) → Linear(hidden_dim → n_actions)
      Critic1: GRU(input_dim → hidden_dim) → Linear(hidden_dim → n_actions)
      Critic2: GRU(input_dim → hidden_dim) → Linear(hidden_dim → n_actions)

    All three GRUs have independent parameters.
    """

    def __init__(self, hidden_dim=64, n_actions=4, input_dim=9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.input_dim = input_dim

        # Actor encoder + head
        self.actor_gru = nn.GRUCell(input_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, n_actions)

        # Twin critic encoders + heads (separate parameters)
        self.critic1_gru = nn.GRUCell(input_dim, hidden_dim)
        self.critic1_head = nn.Linear(hidden_dim, n_actions)

        self.critic2_gru = nn.GRUCell(input_dim, hidden_dim)
        self.critic2_head = nn.Linear(hidden_dim, n_actions)

    def actor_forward(self, x, h_actor):
        """Actor: one GRU step → action logits.

        Args:
            x: (batch, input_dim)
            h_actor: (batch, hidden_dim)

        Returns:
            logits: (batch, n_actions)
            h_new: (batch, hidden_dim)
        """
        h_new = self.actor_gru(x, h_actor)
        logits = self.actor_head(h_new)
        return logits, h_new

    def critic_forward(self, x, h_c1, h_c2):
        """Twin critics: one GRU step → Q-values for all actions.

        Args:
            x: (batch, input_dim)
            h_c1: (batch, hidden_dim)
            h_c2: (batch, hidden_dim)

        Returns:
            q1: (batch, n_actions)
            q2: (batch, n_actions)
            h_c1_new: (batch, hidden_dim)
            h_c2_new: (batch, hidden_dim)
        """
        h_c1_new = self.critic1_gru(x, h_c1)
        q1 = self.critic1_head(h_c1_new)

        h_c2_new = self.critic2_gru(x, h_c2)
        q2 = self.critic2_head(h_c2_new)

        return q1, q2, h_c1_new, h_c2_new

    def init_hidden(self, batch_size=1, device=None):
        """Return zero-initialized hidden states for actor + twin critics."""
        if device is None:
            device = next(self.parameters()).device
        h_a = torch.zeros(batch_size, self.hidden_dim, device=device)
        h_c1 = torch.zeros(batch_size, self.hidden_dim, device=device)
        h_c2 = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h_a, h_c1, h_c2


# =============================================================================
# Episode Replay Buffer
# =============================================================================

class EpisodeReplayBuffer:
    """Stores full episodes and samples random chunks for training.

    Each episode is stored as a dict with:
      - 'inputs': (T, input_dim) tensor
      - 'actions': (T,) tensor of action indices
      - 'rewards': (T,) tensor of float rewards
      - 'length': int
    """

    def __init__(self, capacity=500):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add_episode(self, inputs, actions, rewards):
        """Add one episode to the buffer.

        Args:
            inputs: (T, input_dim) tensor — constructed inputs at each step
            actions: (T,) tensor — action taken at each step
            rewards: (T,) tensor — reward received at each step
        """
        self.buffer.append({
            'inputs': inputs.detach().cpu(),
            'actions': actions.detach().cpu(),
            'rewards': rewards.detach().cpu(),
            'length': len(actions),
        })

    def sample_chunks(self, batch_size, context_len, device='cpu',
                      burn_in=0):
        """Sample random chunks from random episodes.

        When burn_in > 0 (R2D2-style, Kapturowski et al. 2019), returns
        chunks of length (burn_in + context_len).  The first burn_in steps
        are forward-only (warm up hidden states); losses are computed on
        the remaining context_len steps.

        Returns:
            inputs:  (batch_size, burn_in + context_len, input_dim)
            actions: (batch_size, burn_in + context_len)
            rewards: (batch_size, burn_in + context_len)
            masks:   (batch_size, burn_in + context_len)
                     — first burn_in positions have mask=0 (no loss)
        """
        total_len = burn_in + context_len
        input_dim = self.buffer[0]['inputs'].shape[1]
        inputs = torch.zeros(batch_size, total_len, input_dim, device=device)
        actions = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)
        rewards = torch.zeros(batch_size, total_len, device=device)
        masks = torch.zeros(batch_size, total_len, device=device)

        for i in range(batch_size):
            ep = random.choice(self.buffer)
            ep_len = ep['length']

            if ep_len <= total_len:
                # Use entire episode, pad remainder
                inputs[i, :ep_len] = ep['inputs'][:ep_len]
                actions[i, :ep_len] = ep['actions'][:ep_len]
                rewards[i, :ep_len] = ep['rewards'][:ep_len]
                # Mask: only steps after burn_in are trainable
                valid_start = min(burn_in, ep_len)
                masks[i, valid_start:ep_len] = 1.0
            else:
                # Random start within episode
                start = random.randint(0, ep_len - total_len)
                inputs[i] = ep['inputs'][start:start + total_len]
                actions[i] = ep['actions'][start:start + total_len]
                rewards[i] = ep['rewards'][start:start + total_len]
                # Mask: burn_in prefix excluded from loss
                masks[i, burn_in:] = 1.0

        return inputs, actions, rewards, masks

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# Meta-RL SAC Training
# =============================================================================

def _build_input(prev_action, prev_obs, prev_reward, device='cpu'):
    """Build 9-dim input vector from previous step."""
    x = torch.zeros(1, 9, device=device)
    x[0, prev_action] = 1.0
    for j in range(4):
        x[0, 4 + j] = float(prev_obs[j])
    x[0, 8] = prev_reward
    return x


def _collect_episode(net, sim, n_actions, prevent_backward, device='cpu'):
    """Collect one episode using current actor policy.

    Returns:
        inputs: (T, 9) tensor
        actions_t: (T,) tensor
        rewards_t: (T,) tensor
        total_reward: float
    """
    obs = sim.reset()
    h_a, _, _ = net.init_hidden(1, device)

    inputs_list = []
    actions_list = []
    rewards_list = []

    prev_action = 0
    prev_reward = 0.0

    for step in range(n_actions):
        # Build input
        if step == 0:
            x = torch.zeros(1, 9, device=device)
        else:
            x = _build_input(prev_action, obs, prev_reward, device)

        inputs_list.append(x.squeeze(0))

        with torch.no_grad():
            logits, h_a = net.actor_forward(x, h_a)
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
        action = torch.multinomial(probs, 1).item()

        new_obs, reward, moved = sim.step(action)

        actions_list.append(action)
        rewards_list.append(reward)

        prev_action = action
        prev_reward = reward
        obs = new_obs

    inputs_t = torch.stack(inputs_list)
    actions_t = torch.tensor(actions_list, dtype=torch.long)
    rewards_t = torch.tensor(rewards_list, dtype=torch.float32)

    return inputs_t, actions_t, rewards_t, sim.total_reward


def train_meta_rl_sac(
    hidden_dim,
    adj_mats, node_positions_list, start_nodes,
    rewarded_nodes_list, n_actions_list, min_rewarded_states_list,
    n_epochs=100, lr=3e-4, lr_recurrent=None, gamma=0.99, tau=0.005,
    context_len=64, batch_size=32, updates_per_episode=8,
    buffer_capacity=500, grad_clip=1.0, burn_in=0,
    prevent_backward=True, device='cpu', verbose=True,
    training_seed=42,
):
    """Train RecurrentSACNetwork via meta-RL across all maze sessions.

    Args:
        hidden_dim: GRU hidden dimension
        adj_mats: list of adjacency matrices
        node_positions_list: list of node position arrays
        start_nodes: list of start node indices
        rewarded_nodes_list: list of reward arrays
        n_actions_list: list of action counts per session
        min_rewarded_states_list: list of min-reward-states per session
        n_epochs: number of training epochs
        lr: learning rate for head (linear) parameters
        lr_recurrent: learning rate for GRU parameters (default: same as lr).
            BenchNetRL (2025) finds recurrent modules benefit from lower LR
            (e.g. 10× lower than heads).
        gamma: discount factor
        tau: Polyak averaging coefficient
        context_len: chunk length for replay sampling
        batch_size: number of chunks per update
        updates_per_episode: SAC gradient steps after each episode
        buffer_capacity: max episodes in replay buffer
        grad_clip: gradient clipping norm
        burn_in: R2D2-style burn-in steps (Kapturowski et al. 2019).
            Samples chunks of (burn_in + context_len); first burn_in steps
            are forward-only to warm up GRU hidden states, losses computed
            on remaining context_len steps only.
        prevent_backward: mask backward action
        device: torch device
        verbose: print progress
        training_seed: RNG seed for reproducible training (default 42)

    Returns:
        net: trained RecurrentSACNetwork
        epoch_rewards: list of per-epoch average rewards
    """
    # Seed all RNGs for reproducibility
    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)

    n_mazes = len(adj_mats)

    # Online network
    net = RecurrentSACNetwork(hidden_dim=hidden_dim).to(device)

    # Target network (for critic targets only)
    target_net = RecurrentSACNetwork(hidden_dim=hidden_dim).to(device)
    target_net.load_state_dict(net.state_dict())
    for p in target_net.parameters():
        p.requires_grad = False

    # Optimizers — optionally separate LR for recurrent (GRU) vs head params
    lr_rec = lr_recurrent if lr_recurrent is not None else lr

    actor_gru_params = list(net.actor_gru.parameters())
    actor_head_params = list(net.actor_head.parameters())
    actor_optimizer = torch.optim.Adam([
        {'params': actor_gru_params, 'lr': lr_rec},
        {'params': actor_head_params, 'lr': lr},
    ])

    critic_gru_params = (list(net.critic1_gru.parameters()) +
                         list(net.critic2_gru.parameters()))
    critic_head_params = (list(net.critic1_head.parameters()) +
                          list(net.critic2_head.parameters()))
    critic_optimizer = torch.optim.Adam([
        {'params': critic_gru_params, 'lr': lr_rec},
        {'params': critic_head_params, 'lr': lr},
    ])

    # Flat lists for gradient clipping
    actor_params = actor_gru_params + actor_head_params
    critic_params = critic_gru_params + critic_head_params

    # Auto-tuned temperature α
    target_entropy = 0.98 * np.log(4)  # ≈ 1.36
    log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=lr)

    # Replay buffer
    replay = EpisodeReplayBuffer(capacity=buffer_capacity)

    epoch_rewards = []

    for epoch in range(n_epochs):
        total_reward = 0.0
        n_episodes = 0

        # Shuffle session order each epoch
        order = list(range(n_mazes))
        random.shuffle(order)

        for idx in order:
            sim = _EgoSimulator(
                adj_mats[idx], node_positions_list[idx],
                start_nodes[idx], rewarded_nodes_list[idx],
                min_rewarded_states=min_rewarded_states_list[idx],
                prevent_reverse=False,
            )

            # Collect episode with current actor
            inputs_t, actions_t, rewards_t, ep_reward = _collect_episode(
                net, sim, n_actions_list[idx], prevent_backward, device)

            replay.add_episode(inputs_t, actions_t, rewards_t)
            total_reward += ep_reward
            n_episodes += 1

            # SAC gradient updates
            if len(replay) < 2:
                continue

            alpha = log_alpha.exp().detach()

            for _ in range(updates_per_episode):
                # Sample chunks from buffer (with optional burn-in prefix)
                chunk_inputs, chunk_actions, chunk_rewards, chunk_masks = \
                    replay.sample_chunks(batch_size, context_len, device,
                                         burn_in=burn_in)

                total_len = burn_in + context_len

                # ---- Forward pass through GRU sequences ----
                # Initialize hidden states at chunk start (zero-init)
                h_a, h_c1, h_c2 = net.init_hidden(batch_size, device)
                h_tc1 = torch.zeros(batch_size, hidden_dim, device=device)
                h_tc2 = torch.zeros(batch_size, hidden_dim, device=device)

                # Collect per-timestep outputs
                all_logits = []
                all_q1 = []
                all_q2 = []
                all_tgt_q1 = []
                all_tgt_q2 = []
                all_tgt_logits = []

                for t in range(total_len):
                    x_t = chunk_inputs[:, t, :]  # (batch, 9)

                    logits_t, h_a = net.actor_forward(x_t, h_a)
                    q1_t, q2_t, h_c1, h_c2 = net.critic_forward(x_t, h_c1, h_c2)

                    with torch.no_grad():
                        tgt_logits_t, _ = target_net.actor_forward(x_t, h_a.detach())
                        tgt_q1_t, tgt_q2_t, h_tc1, h_tc2 = target_net.critic_forward(
                            x_t, h_tc1, h_tc2)

                    all_logits.append(logits_t)
                    all_q1.append(q1_t)
                    all_q2.append(q2_t)
                    all_tgt_q1.append(tgt_q1_t)
                    all_tgt_q2.append(tgt_q2_t)
                    all_tgt_logits.append(tgt_logits_t)

                # Stack: (batch, total_len, n_actions)
                all_logits = torch.stack(all_logits, dim=1)
                all_q1 = torch.stack(all_q1, dim=1)
                all_q2 = torch.stack(all_q2, dim=1)
                all_tgt_q1 = torch.stack(all_tgt_q1, dim=1)
                all_tgt_q2 = torch.stack(all_tgt_q2, dim=1)
                all_tgt_logits = torch.stack(all_tgt_logits, dim=1)

                # ---- Compute losses for t=0..T-2 (need t+1 for target) ----
                T = total_len - 1

                # Current Q-values for chosen actions
                chosen_q1 = all_q1[:, :T].gather(2, chunk_actions[:, :T].unsqueeze(2)).squeeze(2)
                chosen_q2 = all_q2[:, :T].gather(2, chunk_actions[:, :T].unsqueeze(2)).squeeze(2)

                # Target V(s') = Σ_a π(a|s') [min(Q1_tgt, Q2_tgt) - α·log π(a|s')]
                with torch.no_grad():
                    next_probs = F.softmax(all_tgt_logits[:, 1:T+1], dim=2)
                    next_log_probs = torch.log(next_probs + 1e-8)
                    next_min_q = torch.min(all_tgt_q1[:, 1:T+1], all_tgt_q2[:, 1:T+1])
                    next_v = (next_probs * (next_min_q - alpha * next_log_probs)).sum(dim=2)

                    target_q = chunk_rewards[:, :T] + gamma * next_v

                # Critic loss
                mask_t = chunk_masks[:, :T]
                critic_loss = (mask_t * (chosen_q1 - target_q).pow(2)).sum() / mask_t.sum()
                critic_loss += (mask_t * (chosen_q2 - target_q).pow(2)).sum() / mask_t.sum()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, grad_clip)
                critic_optimizer.step()

                # ---- Actor loss ----
                # Re-forward actor (detach critics)
                h_a2, h_c1_d, h_c2_d = net.init_hidden(batch_size, device)
                actor_logits_all = []
                q1_det_all = []
                q2_det_all = []

                for t in range(total_len):
                    x_t = chunk_inputs[:, t, :]
                    logits_t, h_a2 = net.actor_forward(x_t, h_a2)
                    with torch.no_grad():
                        q1_t, q2_t, h_c1_d, h_c2_d = net.critic_forward(x_t, h_c1_d, h_c2_d)
                    actor_logits_all.append(logits_t)
                    q1_det_all.append(q1_t)
                    q2_det_all.append(q2_t)

                actor_logits_all = torch.stack(actor_logits_all, dim=1)
                q1_det_all = torch.stack(q1_det_all, dim=1)
                q2_det_all = torch.stack(q2_det_all, dim=1)

                probs_all = F.softmax(actor_logits_all, dim=2)
                log_probs_all = torch.log(probs_all + 1e-8)
                min_q_all = torch.min(q1_det_all, q2_det_all)

                # L_π = Σ_a π(a|s)[α·log π(a|s) - Q(s,a)]
                actor_loss = (probs_all * (alpha * log_probs_all - min_q_all)).sum(dim=2)
                actor_loss = (chunk_masks * actor_loss).sum() / chunk_masks.sum()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor_params, grad_clip)
                actor_optimizer.step()

                # ---- Temperature loss ----
                # L_α = -α · [Σ_a π(a|s) log π(a|s) + H_target]
                with torch.no_grad():
                    entropy = -(probs_all * log_probs_all).sum(dim=2)
                    mean_entropy = (chunk_masks * entropy).sum() / chunk_masks.sum()

                alpha_loss = -log_alpha * (mean_entropy - target_entropy)

                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                # ---- Polyak update target networks ----
                with torch.no_grad():
                    for p, p_tgt in zip(net.parameters(), target_net.parameters()):
                        p_tgt.data.mul_(1 - tau).add_(tau * p.data)

        avg_rwd = total_reward / max(n_episodes, 1)
        epoch_rewards.append(avg_rwd)

        if verbose and (epoch + 1) % 5 == 0:
            alpha_val = log_alpha.exp().item()
            print(f"  Epoch {epoch+1}/{n_epochs}: avg_reward={avg_rwd:.3f}  "
                  f"α={alpha_val:.3f}")

    return net, epoch_rewards


# =============================================================================
# RecurrentSACAgent (inference, compatible with run_episode interface)
# =============================================================================

class RecurrentSACAgent:
    """Recurrent SAC agent for maze navigation.

    Uses a trained RecurrentSACNetwork to select actions.
    Compatible with the planning-agent run_episode() interface.

    Args:
        hidden_dim: GRU hidden dimension
        prevent_backward: if True, mask Backward action when other dirs available
        temperature: softmax temperature for action sampling
        weights_path: path to saved model weights (optional)
        model: pre-trained RecurrentSACNetwork (optional, overrides weights_path)
    """

    def __init__(self, hidden_dim=64, prevent_backward=True,
                 temperature=1.0, weights_path=None, model=None):
        self.hidden_dim = hidden_dim
        self.prevent_backward = prevent_backward
        self.temperature = temperature

        if model is not None:
            self.model = model
        elif weights_path is not None:
            self.model = RecurrentSACNetwork(hidden_dim=hidden_dim)
            self.model.load_state_dict(torch.load(weights_path,
                                                   weights_only=True))
        else:
            self.model = RecurrentSACNetwork(hidden_dim=hidden_dim)

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
        reset_interval: int = None,
    ) -> float:
        """Run episode using the trained RecurrentSAC actor policy.

        Parameters
        ----------
        reset_interval : int, optional
            If set, zero-reset the GRU hidden state every *reset_interval*
            steps.  ``reset_interval=1`` makes the agent effectively
            memoryless; ``None`` (default) keeps the hidden state for the
            full episode.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        sim = _EgoSimulator(
            adj_mat, node_positions, start_node, rewarded_nodes,
            min_rewarded_states=min_rewarded_states,
            prevent_reverse=prevent_reverse,
        )
        obs = sim.reset()
        trajectory = [] if return_trajectory else None

        h_a, _, _ = self.model.init_hidden(1)
        prev_action = 0
        prev_reward = 0.0

        for step in range(n_actions):
            # Memory reset ablation: zero the hidden state every K steps
            if reset_interval is not None and step > 0 and step % reset_interval == 0:
                h_a, _, _ = self.model.init_hidden(1)
            # Build input: prev_action_onehot(4) + prev_obs(4) + prev_reward(1)
            x = torch.zeros(1, 9)
            if step > 0:
                x[0, prev_action] = 1.0
                for j in range(4):
                    x[0, 4 + j] = float(obs[j])
                x[0, 8] = prev_reward

            with torch.no_grad():
                logits, h_a = self.model.actor_forward(x, h_a)

            # Apply temperature
            logits = logits.squeeze(0) / self.temperature

            # Get valid ego actions from current observation
            valid_mask = torch.zeros(4)
            for a in range(4):
                if obs[a]:
                    valid_mask[a] = 1.0

            # Optionally mask backward (action 1)
            if self.prevent_backward and valid_mask.sum() > 1 and valid_mask[1] > 0:
                valid_mask[1] = 0.0

            # Mask invalid actions
            logits[valid_mask == 0] = float('-inf')

            # Sample action
            probs = F.softmax(logits, dim=0)
            action = torch.multinomial(probs, 1).item()

            # Execute
            new_obs, reward, moved = sim.step(action)

            if return_trajectory:
                trajectory.append((sim.real_node, reward, 'recurrent_sac'))

            prev_action = action
            prev_reward = reward
            obs = new_obs

        if return_trajectory:
            return sim.total_reward, trajectory
        return sim.total_reward
