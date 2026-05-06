"""
SSM-SAC agent for latent maze navigation.

Replaces GRU in RecurrentSACNetwork with a pure-PyTorch diagonal state-space
model (S4D-style).  No mamba_ssm / CUDA dependency.

Motivation: BenchNetRL (2025) shows Mamba-2 excels at memory-intensive POMDP
tasks (Memory-S11: 0.96 vs GRU: fails), with 3.9× throughput over GRU.
Mamba-2 requires CUDA/Linux; this diagonal SSM provides the SSM architecture
benefit in pure PyTorch for CPU/macOS training.

Architecture:
  Input:  prev_ego_action (one-hot 4) + prev_ego_obs (binary 4) + prev_reward (1) = 9 dims
  Actor:  DiagSSMCell(9 → D) → Linear(D → 4) → softmax → π(a|s)
  Critic1: DiagSSMCell(9 → D) → Linear(D → 4) → Q1(s, a)
  Critic2: DiagSSMCell(9 → D) → Linear(D → 4) → Q2(s, a)

DiagSSMCell implements (S4D, Gu et al. 2022):
  h_{k+1} = A * h_k + B * x_k      (diagonal recurrence, A ∈ (0,1))
  y_k     = C * h_k + D * x_k      (readout)

Where A = sigmoid(log_A_logit) for stable eigenvalues in (0,1).
An optional input-dependent gate (Mamba-style) modulates the output.

References:
  S4D — Gu, Gupta, et al. "On the Parameterization and Initialization of
         Diagonal State Space Models", NeurIPS 2022
  Mamba — Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
          State Spaces", ICLR 2024
  BenchNetRL — "RLBenchNet: Benchmarking Neural Architectures with PPO
               Across RL Tasks", 2025
"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_this_dir = os.path.dirname(os.path.abspath(__file__))
_code_in_mr = os.path.join(_this_dir, '..', '..', 'code')
if _code_in_mr not in sys.path:
    sys.path.insert(0, _code_in_mr)

from intermediate_agents import _EgoSimulator


# =============================================================================
# DiagonalSSMCell
# =============================================================================

class DiagonalSSMCell(nn.Module):
    """Real diagonal state-space model cell (S4D-style).

    Single-step recurrence:
      h_new = A * h + B * x
      y     = C * h_new + D * x

    A = sigmoid(log_A_logit) ∈ (0, 1) for stability.
    Optional gating: y = y * silu(gate(x))  (Mamba-style selection).

    Args:
        input_dim:  dimensionality of input x
        state_dim:  SSM state dimension (like GRU hidden_dim)
        use_gate:   if True, apply Mamba-style input-dependent gating
    """

    def __init__(self, input_dim, state_dim, use_gate=True):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.use_gate = use_gate

        # A: diagonal recurrence coefficients, init near 1 for long memory
        # sigmoid(2.0) ≈ 0.88 — eigenvalues start near 0.88
        self.log_A_logit = nn.Parameter(torch.full((state_dim,), 2.0))

        # B: input → state projection
        self.B = nn.Linear(input_dim, state_dim)

        # C: state → output projection
        self.C = nn.Linear(state_dim, state_dim)

        # D: skip connection
        self.D = nn.Linear(input_dim, state_dim)

        # Optional Mamba-style gate
        if use_gate:
            self.gate = nn.Linear(input_dim, state_dim)

    def forward(self, x, h):
        """Single-step SSM update.

        Args:
            x: (batch, input_dim)
            h: (batch, state_dim) — SSM hidden state

        Returns:
            y: (batch, state_dim) — output (used as hidden for next layer)
            h_new: (batch, state_dim) — updated hidden state
        """
        A = torch.sigmoid(self.log_A_logit)  # (state_dim,) in (0, 1)
        h_new = A * h + self.B(x)            # diagonal recurrence
        y = self.C(h_new) + self.D(x)        # readout + skip

        if self.use_gate:
            y = y * F.silu(self.gate(x))     # input-dependent gating

        return y, h_new


# =============================================================================
# SSMSACNetwork
# =============================================================================

class SSMSACNetwork(nn.Module):
    """SSM-based SAC-Discrete with separate actor and twin-critic SSM encoders.

    Drop-in replacement for RecurrentSACNetwork — same interface.
    Uses DiagonalSSMCell instead of nn.GRUCell.
    """

    def __init__(self, hidden_dim=64, n_actions=4, input_dim=9, use_gate=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.input_dim = input_dim

        # Actor encoder + head
        self.actor_ssm = DiagonalSSMCell(input_dim, hidden_dim, use_gate)
        self.actor_head = nn.Linear(hidden_dim, n_actions)

        # Twin critic encoders + heads (separate parameters)
        self.critic1_ssm = DiagonalSSMCell(input_dim, hidden_dim, use_gate)
        self.critic1_head = nn.Linear(hidden_dim, n_actions)

        self.critic2_ssm = DiagonalSSMCell(input_dim, hidden_dim, use_gate)
        self.critic2_head = nn.Linear(hidden_dim, n_actions)

    def actor_forward(self, x, h_actor):
        """Actor: one SSM step → action logits."""
        y, h_new = self.actor_ssm(x, h_actor)
        logits = self.actor_head(y)
        return logits, h_new

    def critic_forward(self, x, h_c1, h_c2):
        """Twin critics: one SSM step → Q-values for all actions."""
        y1, h_c1_new = self.critic1_ssm(x, h_c1)
        q1 = self.critic1_head(y1)

        y2, h_c2_new = self.critic2_ssm(x, h_c2)
        q2 = self.critic2_head(y2)

        return q1, q2, h_c1_new, h_c2_new

    def init_hidden(self, batch_size=1, device=None):
        """Return zero-initialized hidden states for actor + twin critics."""
        if device is None:
            device = next(self.parameters()).device
        h_a = torch.zeros(batch_size, self.hidden_dim, device=device)
        h_c1 = torch.zeros(batch_size, self.hidden_dim, device=device)
        h_c2 = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h_a, h_c1, h_c2

    # Expose recurrent vs head parameter groups for separate LR
    def recurrent_params(self):
        """SSM cell parameters (lower LR recommended)."""
        return (list(self.actor_ssm.parameters()) +
                list(self.critic1_ssm.parameters()) +
                list(self.critic2_ssm.parameters()))

    def head_params(self):
        """Linear head parameters."""
        return (list(self.actor_head.parameters()) +
                list(self.critic1_head.parameters()) +
                list(self.critic2_head.parameters()))


# =============================================================================
# SSM-SAC Training (reuses recurrent_sac infrastructure)
# =============================================================================

def train_meta_rl_ssm_sac(
    hidden_dim,
    adj_mats, node_positions_list, start_nodes,
    rewarded_nodes_list, n_actions_list, min_rewarded_states_list,
    n_epochs=100, lr=3e-4, lr_recurrent=None, gamma=0.99, tau=0.005,
    context_len=64, batch_size=32, updates_per_episode=8,
    buffer_capacity=500, grad_clip=1.0, burn_in=0,
    use_gate=True,
    prevent_backward=True, device='cpu', verbose=True,
):
    """Train SSMSACNetwork via meta-RL across all maze sessions.

    Same interface as train_meta_rl_sac but uses DiagonalSSMCell.
    """
    from recurrent_sac_agent import EpisodeReplayBuffer, _collect_episode

    n_mazes = len(adj_mats)
    lr_rec = lr_recurrent if lr_recurrent is not None else lr

    # Online network
    net = SSMSACNetwork(hidden_dim=hidden_dim, use_gate=use_gate).to(device)

    # Target network
    target_net = SSMSACNetwork(hidden_dim=hidden_dim, use_gate=use_gate).to(device)
    target_net.load_state_dict(net.state_dict())
    for p in target_net.parameters():
        p.requires_grad = False

    # Optimizers with separate LR for recurrent vs head params
    actor_ssm_params = list(net.actor_ssm.parameters())
    actor_head_params = list(net.actor_head.parameters())
    actor_optimizer = torch.optim.Adam([
        {'params': actor_ssm_params, 'lr': lr_rec},
        {'params': actor_head_params, 'lr': lr},
    ])

    critic_ssm_params = (list(net.critic1_ssm.parameters()) +
                         list(net.critic2_ssm.parameters()))
    critic_head_params = (list(net.critic1_head.parameters()) +
                          list(net.critic2_head.parameters()))
    critic_optimizer = torch.optim.Adam([
        {'params': critic_ssm_params, 'lr': lr_rec},
        {'params': critic_head_params, 'lr': lr},
    ])

    actor_params = actor_ssm_params + actor_head_params
    critic_params = critic_ssm_params + critic_head_params

    # Auto-tuned temperature α
    target_entropy = 0.98 * np.log(4)
    log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=lr)

    # Replay buffer
    replay = EpisodeReplayBuffer(capacity=buffer_capacity)

    epoch_rewards = []

    for epoch in range(n_epochs):
        total_reward = 0.0
        n_episodes = 0

        order = list(range(n_mazes))
        random.shuffle(order)

        for idx in order:
            sim = _EgoSimulator(
                adj_mats[idx], node_positions_list[idx],
                start_nodes[idx], rewarded_nodes_list[idx],
                min_rewarded_states=min_rewarded_states_list[idx],
                prevent_reverse=False,
            )

            # Collect episode — _collect_episode uses net.actor_forward + net.init_hidden
            # which SSMSACNetwork implements with the same interface
            inputs_t, actions_t, rewards_t, ep_reward = _collect_episode(
                net, sim, n_actions_list[idx], prevent_backward, device)

            replay.add_episode(inputs_t, actions_t, rewards_t)
            total_reward += ep_reward
            n_episodes += 1

            if len(replay) < 2:
                continue

            alpha = log_alpha.exp().detach()

            for _ in range(updates_per_episode):
                chunk_inputs, chunk_actions, chunk_rewards, chunk_masks = \
                    replay.sample_chunks(batch_size, context_len, device,
                                         burn_in=burn_in)

                total_len = burn_in + context_len

                # Forward pass through SSM sequences
                h_a, h_c1, h_c2 = net.init_hidden(batch_size, device)
                h_tc1 = torch.zeros(batch_size, hidden_dim, device=device)
                h_tc2 = torch.zeros(batch_size, hidden_dim, device=device)

                all_logits, all_q1, all_q2 = [], [], []
                all_tgt_q1, all_tgt_q2, all_tgt_logits = [], [], []

                for t in range(total_len):
                    x_t = chunk_inputs[:, t, :]

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

                all_logits = torch.stack(all_logits, dim=1)
                all_q1 = torch.stack(all_q1, dim=1)
                all_q2 = torch.stack(all_q2, dim=1)
                all_tgt_q1 = torch.stack(all_tgt_q1, dim=1)
                all_tgt_q2 = torch.stack(all_tgt_q2, dim=1)
                all_tgt_logits = torch.stack(all_tgt_logits, dim=1)

                # Critic loss
                T = total_len - 1
                chosen_q1 = all_q1[:, :T].gather(2, chunk_actions[:, :T].unsqueeze(2)).squeeze(2)
                chosen_q2 = all_q2[:, :T].gather(2, chunk_actions[:, :T].unsqueeze(2)).squeeze(2)

                with torch.no_grad():
                    next_probs = F.softmax(all_tgt_logits[:, 1:T+1], dim=2)
                    next_log_probs = torch.log(next_probs + 1e-8)
                    next_min_q = torch.min(all_tgt_q1[:, 1:T+1], all_tgt_q2[:, 1:T+1])
                    next_v = (next_probs * (next_min_q - alpha * next_log_probs)).sum(dim=2)
                    target_q = chunk_rewards[:, :T] + gamma * next_v

                mask_t = chunk_masks[:, :T]
                critic_loss = (mask_t * (chosen_q1 - target_q).pow(2)).sum() / mask_t.sum()
                critic_loss += (mask_t * (chosen_q2 - target_q).pow(2)).sum() / mask_t.sum()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, grad_clip)
                critic_optimizer.step()

                # Actor loss
                h_a2, h_c1_d, h_c2_d = net.init_hidden(batch_size, device)
                actor_logits_all, q1_det_all, q2_det_all = [], [], []

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

                actor_loss = (probs_all * (alpha * log_probs_all - min_q_all)).sum(dim=2)
                actor_loss = (chunk_masks * actor_loss).sum() / chunk_masks.sum()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor_params, grad_clip)
                actor_optimizer.step()

                # Temperature loss
                with torch.no_grad():
                    entropy = -(probs_all * log_probs_all).sum(dim=2)
                    mean_entropy = (chunk_masks * entropy).sum() / chunk_masks.sum()

                alpha_loss = -log_alpha * (mean_entropy - target_entropy)
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                # Polyak update
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
# SSMSACAgent (inference, compatible with run_episode interface)
# =============================================================================

class SSMSACAgent:
    """SSM-SAC agent for maze navigation.

    Uses a trained SSMSACNetwork to select actions.
    Compatible with the planning-agent run_episode() interface.
    Drop-in replacement for RecurrentSACAgent.
    """

    def __init__(self, hidden_dim=64, prevent_backward=True,
                 temperature=1.0, weights_path=None, model=None,
                 use_gate=True):
        self.hidden_dim = hidden_dim
        self.prevent_backward = prevent_backward
        self.temperature = temperature

        if model is not None:
            self.model = model
        elif weights_path is not None:
            self.model = SSMSACNetwork(hidden_dim=hidden_dim, use_gate=use_gate)
            self.model.load_state_dict(torch.load(weights_path,
                                                   weights_only=True))
        else:
            self.model = SSMSACNetwork(hidden_dim=hidden_dim, use_gate=use_gate)

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
    ) -> float:
        """Run episode using the trained SSM-SAC actor policy."""
        np.random.seed(seed)
        torch.manual_seed(seed)

        sim = _EgoSimulator(
            adj_mat, node_positions, start_node, rewarded_nodes,
            min_rewarded_states=min_rewarded_states,
            prevent_reverse=prevent_reverse,
        )
        obs = sim.reset()

        h_a, _, _ = self.model.init_hidden(1)
        prev_action = 0
        prev_reward = 0.0

        for step in range(n_actions):
            x = torch.zeros(1, 9)
            if step > 0:
                x[0, prev_action] = 1.0
                for j in range(4):
                    x[0, 4 + j] = float(obs[j])
                x[0, 8] = prev_reward

            with torch.no_grad():
                logits, h_a = self.model.actor_forward(x, h_a)

            logits = logits.squeeze(0) / self.temperature

            valid_mask = torch.zeros(4)
            for a in range(4):
                if obs[a]:
                    valid_mask[a] = 1.0

            if self.prevent_backward and valid_mask.sum() > 1 and valid_mask[1] > 0:
                valid_mask[1] = 0.0

            logits[valid_mask == 0] = float('-inf')
            probs = F.softmax(logits, dim=0)
            action = torch.multinomial(probs, 1).item()

            new_obs, reward, moved = sim.step(action)
            prev_action = action
            prev_reward = reward
            obs = new_obs

        return sim.total_reward
