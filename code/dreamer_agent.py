"""
Dreamer-lite agent for latent maze navigation.

Simplified Dreamer (Hafner et al. 2020) for discrete POMDPs with 4-bit
egocentric observations. Learns a latent world model from experience,
then trains actor-critic entirely on imagined rollouts in latent space.

Components:
  1. Encoder: obs(4) → embedding
  2. RSSM: GRU transition model in latent space
  3. Reward head: latent → predicted reward
  4. Actor: latent → action logits (categorical)
  5. Value: latent → scalar value estimate

Training loop (per episode):
  - Collect experience via current policy
  - Encode observed sequences into posterior latent states
  - Train world model (transition + reward prediction)
  - Imagine H-step rollouts from posterior states
  - Compute λ-returns, train actor + value on imagined trajectories

References:
  Hafner et al. 2020, "Dream to Control: Learning Behaviors by Latent
  Imagination", ICLR 2020.
"""

import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

_this_dir = os.path.dirname(os.path.abspath(__file__))
_code_in_mr = os.path.join(_this_dir, '..', '..', 'code')
if _code_in_mr not in sys.path:
    sys.path.insert(0, _code_in_mr)

from intermediate_agents import _EgoSimulator


# ============================================================
# World Model
# ============================================================

class WorldModel(nn.Module):
    """Latent world model: encoder + GRU transition + reward head.

    The RSSM uses a deterministic GRU (no stochastic component) for
    simplicity — our observations are so low-dimensional (4 bits) that
    the posterior ≈ prior most of the time.
    """

    def __init__(self, obs_dim=4, n_actions=4, latent_dim=32, embed_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions

        # Encoder: obs → embedding
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
        )

        # Posterior: incorporate observation into latent state
        # Input: embedding + action_onehot
        self.posterior_gru = nn.GRUCell(embed_dim + n_actions, latent_dim)

        # Prior (transition): predict next latent from current + action
        # Same GRU but without observation (for imagination)
        self.prior_gru = nn.GRUCell(n_actions, latent_dim)

        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Observation predictor (for world model loss)
        self.obs_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, obs_dim),
        )

    def encode_step(self, obs, action_onehot, h):
        """Posterior update: incorporate observation into latent state."""
        e = self.obs_embed(obs)
        inp = torch.cat([e, action_onehot], dim=-1)
        h_new = self.posterior_gru(inp, h)
        return h_new

    def imagine_step(self, action_onehot, h):
        """Prior transition: predict next latent without observation."""
        h_new = self.prior_gru(action_onehot, h)
        return h_new

    def predict_reward(self, h):
        return self.reward_head(h).squeeze(-1)

    def predict_obs(self, h):
        return self.obs_head(h)

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.latent_dim)


# ============================================================
# Actor-Critic
# ============================================================

class ActorCritic(nn.Module):
    """Actor and value networks operating on latent states."""

    def __init__(self, latent_dim=32, n_actions=4):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def policy(self, h):
        """Returns action logits."""
        return self.actor(h)

    def value(self, h):
        """Returns scalar value estimate."""
        return self.critic(h).squeeze(-1)


# ============================================================
# Dreamer-Lite Agent
# ============================================================

class DreamerLiteAgent:
    """Dreamer-lite for discrete POMDP maze navigation.

    Single-episode online learning: collects experience, trains world
    model + actor-critic, then uses the trained policy.

    This is a "within-episode" Dreamer: it learns from the ongoing
    episode's experience and imagines forward to train the policy.
    More similar to online model-based RL than full Dreamer's replay.

    Parameters:
        latent_dim: GRU hidden state dimension
        embed_dim: observation embedding dimension
        horizon: imagination rollout length
        gamma: discount factor
        lam: λ for GAE/λ-returns
        lr_world: world model learning rate
        lr_actor: actor learning rate
        lr_critic: critic learning rate
        train_interval: train every N steps
        train_epochs: gradient steps per training call
        n_imagine: number of imagination rollouts per training call
        prevent_backward: mask backward action when possible
    """

    def __init__(self, latent_dim=32, embed_dim=16, horizon=15,
                 gamma=0.99, lam=0.95,
                 lr_world=3e-3, lr_actor=1e-3, lr_critic=1e-3,
                 train_interval=16, train_epochs=4, n_imagine=8,
                 prevent_backward=True):
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.lr_world = lr_world
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.train_interval = train_interval
        self.train_epochs = train_epochs
        self.n_imagine = n_imagine
        self.prevent_backward = prevent_backward

        self.world_model = WorldModel(latent_dim=latent_dim, embed_dim=embed_dim)
        self.actor_critic = ActorCritic(latent_dim=latent_dim)

        self.opt_world = torch.optim.Adam(self.world_model.parameters(), lr=lr_world)
        self.opt_actor = torch.optim.Adam(self.actor_critic.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.actor_critic.critic.parameters(), lr=lr_critic)

        # Experience buffer (lists, converted to tensors for training)
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []

    def _action_onehot(self, action, batch=False):
        if batch:
            oh = torch.zeros(len(action), 4)
            for i, a in enumerate(action):
                oh[i, a] = 1.0
            return oh
        oh = torch.zeros(1, 4)
        oh[0, action] = 1.0
        return oh

    def _train_world_model(self):
        """Train world model on collected experience."""
        if len(self.obs_buf) < 4:
            return

        obs = torch.FloatTensor(np.array(self.obs_buf))
        acts = self.act_buf
        rews = torch.FloatTensor(self.rew_buf)

        T = len(obs)
        h = self.world_model.init_hidden(1)

        total_loss = torch.tensor(0.0)
        for t in range(T - 1):
            a_oh = self._action_onehot(acts[t])
            h = self.world_model.encode_step(obs[t:t+1], a_oh, h)

            # Predict next observation and reward
            pred_obs = self.world_model.predict_obs(h)
            pred_rew = self.world_model.predict_reward(h)

            obs_loss = F.binary_cross_entropy_with_logits(pred_obs, obs[t+1:t+2])
            rew_loss = F.mse_loss(pred_rew, rews[t+1:t+2])
            total_loss = total_loss + obs_loss + rew_loss

            h = h.detach()  # Truncated BPTT

        if T > 2:
            total_loss = total_loss / (T - 1)
            self.opt_world.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.world_model.parameters(), 10.0)
            self.opt_world.step()

    def _train_actor_critic(self):
        """Train actor-critic on imagined rollouts from recent states."""
        if len(self.obs_buf) < 4:
            return

        obs = torch.FloatTensor(np.array(self.obs_buf))
        acts = self.act_buf
        T = len(obs)

        # Encode experience to get posterior latent states
        with torch.no_grad():
            h = self.world_model.init_hidden(1)
            latent_states = []
            for t in range(T):
                a_oh = self._action_onehot(acts[t] if t < len(acts) else 0)
                h = self.world_model.encode_step(obs[t:t+1], a_oh, h)
                latent_states.append(h)

        # Sample starting states for imagination
        n_starts = min(self.n_imagine, T)
        start_indices = np.random.choice(T, n_starts, replace=False)
        start_h = torch.cat([latent_states[i] for i in start_indices], dim=0)

        # Imagine forward
        H = min(self.horizon, 30)
        imagined_h = [start_h]
        imagined_actions = []
        imagined_rewards = []

        h_im = start_h
        for t in range(H):
            logits = self.actor_critic.policy(h_im)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            imagined_actions.append(action)

            a_oh = torch.zeros(n_starts, 4)
            a_oh.scatter_(1, action.unsqueeze(-1), 1.0)

            h_im = self.world_model.imagine_step(a_oh, h_im)
            r_im = self.world_model.predict_reward(h_im)

            imagined_h.append(h_im)
            imagined_rewards.append(r_im)

        if len(imagined_rewards) == 0:
            return

        # Compute λ-returns
        rewards = torch.stack(imagined_rewards)  # (H, n_starts)
        values = torch.stack([self.actor_critic.value(h) for h in imagined_h])  # (H+1, n_starts)

        # GAE λ-returns
        returns = torch.zeros_like(rewards)
        last_val = values[-1].detach()
        last_gae = torch.zeros(n_starts)

        for t in reversed(range(H)):
            delta = rewards[t] + self.gamma * values[t+1].detach() - values[t].detach()
            last_gae = delta + self.gamma * self.lam * last_gae
            returns[t] = last_gae + values[t].detach()

        # Actor loss: maximize imagined returns
        log_probs = []
        h_im = start_h
        for t in range(H):
            logits = self.actor_critic.policy(h_im)
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(imagined_actions[t])
            log_probs.append(lp)

            a_oh = torch.zeros(n_starts, 4)
            a_oh.scatter_(1, imagined_actions[t].unsqueeze(-1), 1.0)
            h_im = self.world_model.imagine_step(a_oh, h_im)

        log_probs = torch.stack(log_probs)  # (H, n_starts)
        advantages = (returns - values[:-1].detach())
        actor_loss = -(log_probs * advantages.detach()).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), 10.0)
        self.opt_actor.step()

        # Critic loss: regress values to λ-returns
        value_preds = values[:-1]
        critic_loss = F.mse_loss(value_preds, returns.detach())

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), 10.0)
        self.opt_critic.step()

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
        """Run a single episode with online Dreamer learning."""
        np.random.seed(seed)
        torch.manual_seed(seed)

        use_pr = prevent_reverse or self.prevent_backward
        sim = _EgoSimulator(
            adj_mat, node_positions, start_node, rewarded_nodes,
            min_rewarded_states=min_rewarded_states,
            prevent_reverse=use_pr,
        )
        obs = sim.reset()

        # Reset buffers and models for fresh episode
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []

        h = self.world_model.init_hidden(1)
        prev_action = 0

        for step in range(n_actions):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)

            # Encode current observation
            a_oh = self._action_onehot(prev_action)
            with torch.no_grad():
                h = self.world_model.encode_step(obs_t, a_oh, h)

            # Select action from policy
            with torch.no_grad():
                logits = self.actor_critic.policy(h)

            # Mask invalid actions
            valid_mask = torch.zeros(4)
            for a in range(4):
                if obs[a]:
                    valid_mask[a] = 1.0
            if use_pr and valid_mask.sum() > 1 and valid_mask[1] > 0:
                valid_mask[1] = 0.0
            logits = logits.squeeze(0)
            logits[valid_mask == 0] = float('-inf')

            # Epsilon-greedy exploration (decays over episode)
            eps = max(0.05, 1.0 - step / max(n_actions * 0.5, 1))
            if np.random.random() < eps:
                valid_actions = torch.where(valid_mask > 0)[0]
                action = valid_actions[np.random.randint(len(valid_actions))].item()
            else:
                probs = F.softmax(logits, dim=0)
                action = torch.multinomial(probs, 1).item()

            # Execute
            new_obs, reward, moved = sim.step(action)

            # Store experience
            self.obs_buf.append(obs)
            self.act_buf.append(action)
            self.rew_buf.append(reward)

            prev_action = action
            obs = new_obs

            # Periodic training
            if (step + 1) % self.train_interval == 0 and step > self.train_interval:
                for _ in range(self.train_epochs):
                    self._train_world_model()
                    self._train_actor_critic()

        return sim.total_reward
