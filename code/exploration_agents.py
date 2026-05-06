"""
Exploration-driven RL agents for graph-based maze navigation.

Agents:

  MOP (Maximum Occupancy Principle) — Planning agent
    Faithful implementation of Ramírez-Ruiz, Grytskyy, Bhatt & Moreno-Bote,
    "Complex behavior from intrinsic motivation to occupy action-state path
    space", Nature Communications 2024.
    Tabular value iteration on the graph adjacency matrix. Optimal policy
    maximizes discounted path entropy: π*(a|s) ∝ exp(γ·V*(s')/α).
    For deterministic environments (our case), β drops out entirely.
    Uses the planning agent API (run_episode), not the SB3 API.
    Extension: optional reward_weight parameter blends extrinsic reward
    into value iteration, interpolating between pure MOP and reward-seeking.

  RC-GVF (Random Cumulant - General Value Functions)
    Ensemble of GVF heads predicts fixed random pseudo-rewards.
    Intrinsic reward = ensemble disagreement (std across heads).
    Follows the DRQNAgent API (SB3-compatible, learn()/set_env()).

  CountBasedExplorerAgent (SHELVED — formerly "MOPagent")
    SimHash + 1/sqrt(count) novelty with A2C. Does NOT implement the
    Moreno-Bote MOP. Kept for reproducibility.

References:
  MOP — Ramírez-Ruiz et al., Nature Communications 2024
  RC-GVF — Lanclos et al., NeurIPS 2022
"""

import os
import sys
import random
from copy import deepcopy

import numpy as np
from scipy.special import logsumexp

# Add code to path for utils imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_code_in_mr = os.path.join(_this_dir, '..', '..', 'code')
if _code_in_mr not in sys.path:
    sys.path.insert(0, _code_in_mr)

from utils_latMaz import get_adj_states


# =================================================================
# MOP Agent — Tabular Maximum Occupancy Principle (Planning Agent)
# =================================================================

class MOPagent:
    """Maximum Occupancy Principle agent (Ramírez-Ruiz et al., 2024).

    For deterministic graph-based mazes, MOP reduces to:
      V*(s) = α · ln[ Σ_{a∈A(s)} exp(γ · V*(T(s,a)) / α) ]
      π*(a|s) = exp(γ · V*(s') / α) / Z(s)

    where T(s,a) is the deterministic transition function, and the
    state-transition entropy H(S'|s,a)=0 since transitions are
    deterministic (β drops out).

    The agent computes V* via value iteration on the adjacency matrix,
    then samples actions from the softmax MOP policy.

    With reward_weight=0 (default), this is pure MOP — the agent is
    entirely reward-blind and collects rewards only incidentally.
    With reward_weight > 0, the value iteration blends in extrinsic
    reward, making the agent reward-seeking:
      V*(s) = α · ln[ Σ exp((γ·V*(s') + w·R(s')) / α) ]

    Parameters:
        alpha: Temperature / action entropy weight (default 1.0).
        gamma_mop: Discount factor for MOP value iteration (default 0.99).
            Named gamma_mop to avoid collision with RL gamma in config.
        max_iter: Max value iteration steps (default 500).
        tol: Convergence tolerance for value iteration (default 1e-6).
        reward_weight: Weight for extrinsic reward in value iteration
            (default 0.0 = pure MOP). When > 0, adds w * R(s') to the
            exponent, making the agent reward-seeking.
    """

    def __init__(self, alpha=1.0, gamma_mop=0.99, max_iter=500, tol=1e-6,
                 reward_weight=0.0):
        self.alpha = alpha
        self.gamma_mop = gamma_mop
        self.max_iter = max_iter
        self.tol = tol
        self.reward_weight = reward_weight

    @staticmethod
    def _largest_component(adj_mat):
        """Return boolean mask of nodes in the largest connected component.

        Wall / inaccessible nodes (degree 0) are excluded.  Among
        remaining nodes, BFS finds all connected components and the
        largest one wins.
        """
        from collections import deque
        N = adj_mat.shape[0]
        visited = np.zeros(N, dtype=bool)
        components = []

        for start in range(N):
            if visited[start]:
                continue
            nbrs = np.where(adj_mat[start] > 0)[0]
            if len(nbrs) == 0:
                visited[start] = True
                continue  # degree-0 wall node
            component = []
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                for n in np.where(adj_mat[node] > 0)[0]:
                    if not visited[n]:
                        queue.append(n)
            if component:
                components.append(component)

        if not components:
            return np.zeros(N, dtype=bool)
        largest = max(components, key=len)
        mask = np.zeros(N, dtype=bool)
        mask[largest] = True
        return mask

    @staticmethod
    def _build_vi_cache(adj_mat):
        """Precompute padded neighbor index matrix + mask for vectorized VI."""
        N = adj_mat.shape[0]
        adj = (adj_mat > 0)
        max_deg = max(int(adj.sum(axis=1).max()), 1)
        nbr_idx = np.zeros((N, max_deg), dtype=int)
        nbr_mask = np.full((N, max_deg), -np.inf)
        for s in range(N):
            nbrs = np.where(adj[s])[0]
            k = len(nbrs)
            if k > 0:
                nbr_idx[s, :k] = nbrs
                nbr_mask[s, :k] = 0.0
        return nbr_idx, nbr_mask

    def _value_iteration(self, nbr_idx, nbr_mask, component_mask,
                         rewards=None, V_init=None, max_iter=None):
        """Compute MOP optimal values V*(s) for all nodes (vectorized).

        Uses the iterative fixed-point equation (Eq. 7 from the paper):
          V*(s) = α · ln[ Σ_{s'∈N(s)} exp((γ·V*(s') + w·R(s')) / α) ]

        Nodes outside the largest connected component get NaN so that
        callers can distinguish walls / isolated fragments.

        Args:
            nbr_idx, nbr_mask: from _build_vi_cache()
            component_mask: boolean mask from _largest_component()
            rewards: current reward vector (optional, used when reward_weight > 0)
            V_init: warm-start values (optional, avoids cold-start after rebait)
            max_iter: override self.max_iter (for warm-started incremental updates)
        """
        N = nbr_idx.shape[0]
        if V_init is not None:
            V = V_init.copy()
        else:
            V = np.full(N, np.nan)
            V[component_mask] = 0.0

        mi = max_iter if max_iter is not None else self.max_iter

        # Reward bonus per node
        r_bonus = np.zeros(N)
        if rewards is not None and self.reward_weight > 0:
            r_bonus = self.reward_weight * np.array(rewards, dtype=float)

        inv_alpha = 1.0 / self.alpha
        for iteration in range(mi):
            V_old = V.copy()
            V_nbrs = V_old[nbr_idx]
            R_nbrs = r_bonus[nbr_idx]
            exponents = (self.gamma_mop * V_nbrs + R_nbrs) * inv_alpha + nbr_mask
            V_new = self.alpha * logsumexp(exponents, axis=1)
            # Only update component nodes; wall/isolated nodes stay NaN
            V = np.where(component_mask, V_new, np.nan)

            comp_diff = np.abs(V[component_mask] - V_old[component_mask])
            if np.max(comp_diff) < self.tol:
                break

        return V

    def compute_values(self, adj_mat, rewards=None):
        """Convenience: compute V*(s) from an adjacency matrix.

        Returns V with NaN for wall/isolated nodes.
        """
        component_mask = self._largest_component(adj_mat)
        nbr_idx, nbr_mask = self._build_vi_cache(adj_mat)
        return self._value_iteration(nbr_idx, nbr_mask, component_mask,
                                     rewards=rewards)

    def _mop_policy(self, node, V_star, neighbors, rewards=None):
        """Compute MOP action probabilities over neighbors of node.

        π*(neighbor|node) = exp((γ·V*(neighbor) + w·R(neighbor)) / α) / Z(node)
        """
        r_bonus = np.zeros(len(neighbors))
        if rewards is not None and self.reward_weight > 0:
            r_bonus = self.reward_weight * np.array([rewards[n] for n in neighbors])
        logits = (self.gamma_mop * V_star[neighbors] + r_bonus) / self.alpha
        # Exclude neighbors outside the component (NaN V*) → -inf logit
        logits = np.where(np.isnan(logits), -np.inf, logits)
        logits -= np.nanmax(logits)  # numerical stability
        probs = np.exp(logits)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            # Fallback: uniform over all neighbors
            probs = np.ones(len(neighbors)) / len(neighbors)
        return probs

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
        discover_rewards: bool = False,
    ) -> float:
        """Run a single MOP episode on the graph maze.

        The agent follows the MOP policy (entropy-maximizing, optionally
        reward-biased). Rewards are collected when visiting rewarded nodes,
        using the same rebait logic as GraphMazeEnv (CHECK → REBAIT → REMOVE).

        Args:
            discover_rewards: If True, agent starts with NO reward knowledge
                and discovers rewards only by visiting nodes. Only relevant
                when reward_weight > 0 (pure MOP ignores rewards regardless).

        Returns:
            Total reward collected during the episode.
        """
        np.random.seed(seed)
        random.seed(seed)

        n_nodes = len(adj_mat)
        neighbor_list = [np.where(adj_mat[s] > 0)[0] for s in range(n_nodes)]

        # Wall-node exclusion + vectorized VI cache
        component_mask = self._largest_component(adj_mat)
        nbr_idx, nbr_mask = self._build_vi_cache(adj_mat)

        # Ground-truth reward state (what the environment actually has)
        env_rwd = np.array(rewarded_nodes, dtype=float)
        env_rwd[~component_mask] = 0.0  # zero out unreachable nodes
        env_rwd_initial = env_rwd.copy()

        # Agent's reward belief (what the agent plans with)
        if discover_rewards:
            agent_rwd = np.zeros(n_nodes, dtype=float)
            discovered_rewarded = set()
        else:
            agent_rwd = env_rwd.copy()

        # Compute MOP optimal values (NaN for wall / non-component nodes)
        V_star = self._value_iteration(nbr_idx, nbr_mask, component_mask,
                                       agent_rwd)

        current = start_node
        prev_node = None
        total_reward = 0.0

        for step in range(n_actions):
            nbrs = neighbor_list[current]
            if len(nbrs) == 0:
                break

            # Filter out previous node if prevent_reverse is on
            # (allow at dead-ends where there's only one neighbor)
            if prevent_reverse and prev_node is not None and len(nbrs) > 1:
                nbrs = nbrs[nbrs != prev_node]

            # MOP policy: softmax over neighbor values (+reward bonus)
            probs = self._mop_policy(current, V_star, nbrs, agent_rwd)
            next_node = np.random.choice(nbrs, p=probs)

            prev_node = current
            current = next_node

            # Collect reward (CHECK → REBAIT → REMOVE, matching fixed GraphMazeEnv)
            if env_rwd[current] > 0:
                total_reward += env_rwd[current]
                # Check rebait threshold BEFORE removal
                did_rebait = False
                if int(np.sum(env_rwd > 0)) == min_rewarded_states:
                    env_rwd = env_rwd_initial.copy()  # REBAIT
                    did_rebait = True
                env_rwd[current] = 0.0  # THEN REMOVE

                # Update agent's reward belief
                if discover_rewards:
                    discovered_rewarded.add(current)
                    if did_rebait:
                        for nd in discovered_rewarded:
                            agent_rwd[nd] = 1.0
                        agent_rwd[current] = 0.0
                    else:
                        agent_rwd[current] = 0.0
                else:
                    if did_rebait:
                        agent_rwd = env_rwd.copy()
                    else:
                        agent_rwd[current] = 0.0

                # Warm-start VI recomputation after reward map change
                if self.reward_weight > 0:
                    V_star = self._value_iteration(
                        nbr_idx, nbr_mask, component_mask, agent_rwd,
                        V_init=V_star, max_iter=50)

        return total_reward

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    from gymnasium import spaces

    # =================================================================
    # Shared GRU encoder
    # =================================================================

    class SharedGRUEncoder(nn.Module):
        """obs(4) -> Linear -> ReLU -> GRU -> encoded hidden state.

        Maintains hidden state across steps within an episode.
        """

        def __init__(self, obs_dim=4, hidden_dim=64):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.embed = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
            )
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        def forward(self, obs, hidden):
            """
            Args:
                obs: (batch, obs_dim) single step or (batch, seq, obs_dim)
                hidden: (1, batch, hidden_dim) or None
            Returns:
                encoded: (batch, hidden_dim)
                new_hidden: (1, batch, hidden_dim)
            """
            x = self.embed(obs)
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
            out, new_hidden = self.gru(x, hidden)
            # Return last time-step encoding
            encoded = out[:, -1, :]  # (batch, hidden_dim)
            return encoded, new_hidden

        def init_hidden(self, batch_size=1):
            return torch.zeros(1, batch_size, self.hidden_dim)

    # =================================================================
    # RC-GVF Agent
    # =================================================================

    class RCGVFAgent:
        """Random Cumulant GVF agent with ensemble disagreement exploration.

        Architecture:
          SharedGRUEncoder -> policy_head (A2C actor)
                           -> value_head  (A2C critic)
          random_cumulant_net (FIXED weights) -> pseudo-rewards Z
          gvf_ensemble (N heads) -> predict GVF values from encoder hidden

        Intrinsic reward = std(ensemble_predictions).mean()
        Policy trained with A2C at end of episode using combined reward.
        """

        def __init__(self, policy_str, env, verbose=0, device="cpu",
                     seed=0, **kwargs):
            self.env = env
            self.verbose = verbose
            self.device = torch.device(device)
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Hyperparameters
            self.hidden_dim = kwargs.get('hidden_dim', 64)
            self.n_cumulants = kwargs.get('n_cumulants', 8)
            self.n_ensemble = kwargs.get('n_ensemble', 5)
            self.intrinsic_coef = kwargs.get('intrinsic_coef', 0.1)
            self.gvf_gamma = kwargs.get('gvf_gamma', 0.9)
            self.lr = kwargs.get('learning_rate', 1e-3)
            self.gamma = kwargs.get('gamma', 0.99)
            self.ent_coef = kwargs.get('ent_coef', 0.01)
            self.vf_coef = kwargs.get('vf_coef', 0.5)
            self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)

            # Derive dims from env
            obs_sp = env.observation_space
            if isinstance(obs_sp, spaces.MultiBinary):
                self.obs_dim = obs_sp.n
            elif isinstance(obs_sp, spaces.Box):
                self.obs_dim = obs_sp.shape[0]
            else:
                self.obs_dim = obs_sp.n
            self.n_actions = env.action_space.n

            # Networks
            self.encoder = SharedGRUEncoder(self.obs_dim, self.hidden_dim).to(self.device)
            self.policy_head = nn.Linear(self.hidden_dim, self.n_actions).to(self.device)
            self.value_head = nn.Linear(self.hidden_dim, 1).to(self.device)

            # Random cumulant network (FIXED — never trained)
            self.random_cumulant_net = nn.Sequential(
                nn.Linear(self.obs_dim, 32),
                nn.ReLU(),
                nn.Linear(32, self.n_cumulants),
            ).to(self.device)
            for p in self.random_cumulant_net.parameters():
                p.requires_grad = False

            # GVF ensemble: N heads predicting K cumulant values from encoding
            self.gvf_ensemble = nn.ModuleList([
                nn.Linear(self.hidden_dim, self.n_cumulants)
                for _ in range(self.n_ensemble)
            ]).to(self.device)

            # Optimiser covers encoder + policy + value + gvf_ensemble
            self.optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) +
                list(self.policy_head.parameters()) +
                list(self.value_head.parameters()) +
                list(self.gvf_ensemble.parameters()),
                lr=self.lr,
            )

        def _get_intrinsic_reward(self, encoded):
            """Ensemble disagreement: std across N head predictions."""
            preds = torch.stack([head(encoded) for head in self.gvf_ensemble], dim=0)
            # preds: (N, batch, K)
            return preds.std(dim=0).mean(dim=-1)  # (batch,)

        def set_env(self, new_env):
            """Swap environment. Keeps network weights."""
            self.env = new_env

        def learn(self, total_timesteps=None, callback=None, **kwargs):
            """Run one episode with online GVF updates, end-of-episode A2C."""
            obs, _ = self.env.reset()
            hidden = self.encoder.init_hidden(1).to(self.device)

            # Episode storage
            log_probs = []
            values = []
            rewards_ext = []
            rewards_int = []
            entropies = []

            # GVF training storage
            gvf_encodings = []
            gvf_cumulants = []
            gvf_next_encodings = []

            done = False
            prev_encoded = None
            while not done:
                obs_t = torch.FloatTensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    cumulant = self.random_cumulant_net(obs_t)  # (1, K)

                encoded, hidden = self.encoder(obs_t, hidden)

                # Intrinsic reward from ensemble disagreement
                with torch.no_grad():
                    r_int = self._get_intrinsic_reward(encoded).item()

                # Policy
                logits = self.policy_head(encoded)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                # Value
                value = self.value_head(encoded).squeeze(-1)

                # Store for GVF learning
                if prev_encoded is not None:
                    gvf_encodings.append(prev_encoded.detach())
                    gvf_cumulants.append(prev_cumulant.detach())
                    gvf_next_encodings.append(encoded.detach())
                prev_encoded = encoded
                prev_cumulant = cumulant

                # Step env
                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                log_probs.append(log_prob)
                values.append(value)
                rewards_ext.append(float(reward))
                rewards_int.append(r_int)
                entropies.append(entropy)

                obs = next_obs

            # -- End of episode: A2C policy/value update --
            T = len(rewards_ext)
            if T == 0:
                return

            # Combine rewards
            combined = [
                rewards_ext[t] + self.intrinsic_coef * rewards_int[t]
                for t in range(T)
            ]

            # Compute returns (no bootstrapping — episode ended)
            returns = []
            G = 0.0
            for r in reversed(combined):
                G = r + self.gamma * G
                returns.insert(0, G)

            returns_t = torch.FloatTensor(returns).to(self.device)
            values_t = torch.stack(values).squeeze()
            log_probs_t = torch.stack(log_probs).squeeze()
            entropies_t = torch.stack(entropies).squeeze()

            advantages = returns_t - values_t.detach()
            policy_loss = -(log_probs_t * advantages).mean()
            value_loss = F.mse_loss(values_t, returns_t)
            entropy_loss = -entropies_t.mean()

            # -- GVF ensemble TD loss --
            gvf_loss = torch.tensor(0.0, device=self.device)
            if gvf_encodings:
                enc_batch = torch.cat(gvf_encodings, dim=0)       # (T-1, hidden_dim)
                cum_batch = torch.cat(gvf_cumulants, dim=0)       # (T-1, K)
                next_enc_batch = torch.cat(gvf_next_encodings, dim=0)

                # Target: Z(obs_next) + gvf_gamma * mean_ensemble(h_next)
                with torch.no_grad():
                    mean_next = torch.stack(
                        [head(next_enc_batch) for head in self.gvf_ensemble], dim=0
                    ).mean(dim=0)
                    gvf_target = cum_batch + self.gvf_gamma * mean_next

                for head in self.gvf_ensemble:
                    pred = head(enc_batch)
                    gvf_loss = gvf_loss + F.mse_loss(pred, gvf_target)
                gvf_loss = gvf_loss / self.n_ensemble

            total_loss = (
                policy_loss
                + self.vf_coef * value_loss
                + self.ent_coef * entropy_loss
                + gvf_loss
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.policy_head.parameters()) +
                list(self.value_head.parameters()) +
                list(self.gvf_ensemble.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

    # =================================================================
    # SHELVED: Count-based "MOP" Agent (does NOT match Moreno-Bote MOP)
    # =================================================================
    # This agent uses SimHash count-based novelty (1/sqrt(count)) with A2C.
    # It does NOT implement the Maximum Occupancy Principle from:
    #   Ramírez-Ruiz et al., "Complex behavior from intrinsic motivation
    #   to occupy action-state path space", Nature Communications 2024.
    # The paper's MOP maximizes discounted action-state PATH ENTROPY,
    # not count-based state novelty. See MOPagent (below) for the
    # faithful reimplementation.
    # Kept for reproducibility of earlier experiment results.
    # =================================================================

    class CountBasedExplorerAgent:
        """SHELVED — formerly "MOPagent" but does NOT match Moreno-Bote MOP.

        This is a count-based novelty agent (SimHash + 1/sqrt(count) + A2C).
        Renamed to avoid confusion with the actual Maximum Occupancy Principle.

        Architecture:
          SharedGRUEncoder -> policy_head (A2C actor)
                           -> value_head  (A2C critic)
          SimHash of GRU hidden state -> count table -> 1/sqrt(count)

        Intrinsic reward = 1/sqrt(visit_count(hash(h_t)))
        Policy trained with A2C at end of episode using combined reward.
        """

        def __init__(self, policy_str, env, verbose=0, device="cpu",
                     seed=0, **kwargs):
            self.env = env
            self.verbose = verbose
            self.device = torch.device(device)
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Hyperparameters
            self.hidden_dim = kwargs.get('hidden_dim', 64)
            self.hash_dim = kwargs.get('hash_dim', 32)
            self.intrinsic_coef = kwargs.get('intrinsic_coef', 0.1)
            self.lr = kwargs.get('learning_rate', 1e-3)
            self.gamma = kwargs.get('gamma', 0.99)
            self.ent_coef = kwargs.get('ent_coef', 0.01)
            self.vf_coef = kwargs.get('vf_coef', 0.5)
            self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)

            # Derive dims from env
            obs_sp = env.observation_space
            if isinstance(obs_sp, spaces.MultiBinary):
                self.obs_dim = obs_sp.n
            elif isinstance(obs_sp, spaces.Box):
                self.obs_dim = obs_sp.shape[0]
            else:
                self.obs_dim = obs_sp.n
            self.n_actions = env.action_space.n

            # Networks
            self.encoder = SharedGRUEncoder(self.obs_dim, self.hidden_dim).to(self.device)
            self.policy_head = nn.Linear(self.hidden_dim, self.n_actions).to(self.device)
            self.value_head = nn.Linear(self.hidden_dim, 1).to(self.device)

            # SimHash: fixed random projection for count-based hashing
            self._hash_matrix = torch.randn(self.hidden_dim, self.hash_dim, device=self.device)
            self._visit_counts = {}

            # Optimiser
            self.optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) +
                list(self.policy_head.parameters()) +
                list(self.value_head.parameters()),
                lr=self.lr,
            )

        def _hash_state(self, encoded):
            """SimHash: sign(h @ random_matrix) -> binary tuple key."""
            with torch.no_grad():
                proj = encoded @ self._hash_matrix  # (1, hash_dim)
                bits = (proj > 0).cpu().numpy().flatten()
            return tuple(bits.astype(int))

        def _get_intrinsic_reward(self, encoded):
            """Count-based novelty: 1/sqrt(count(hash(h)))."""
            key = self._hash_state(encoded)
            self._visit_counts[key] = self._visit_counts.get(key, 0) + 1
            return 1.0 / np.sqrt(self._visit_counts[key])

        def set_env(self, new_env):
            """Swap environment. Keeps weights, resets visit counts."""
            self.env = new_env
            self._visit_counts = {}

        def learn(self, total_timesteps=None, callback=None, **kwargs):
            """Run one episode with count-based exploration, end-of-episode A2C."""
            obs, _ = self.env.reset()
            hidden = self.encoder.init_hidden(1).to(self.device)

            # Episode storage
            log_probs = []
            values = []
            rewards_ext = []
            rewards_int = []
            entropies = []

            done = False
            while not done:
                obs_t = torch.FloatTensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)

                encoded, hidden = self.encoder(obs_t, hidden)

                # Intrinsic reward (count-based)
                r_int = self._get_intrinsic_reward(encoded)

                # Policy
                logits = self.policy_head(encoded)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                # Value
                value = self.value_head(encoded).squeeze(-1)

                # Step env
                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                log_probs.append(log_prob)
                values.append(value)
                rewards_ext.append(float(reward))
                rewards_int.append(r_int)
                entropies.append(entropy)

                obs = next_obs

            # -- End of episode: A2C policy/value update --
            T = len(rewards_ext)
            if T == 0:
                return

            combined = [
                rewards_ext[t] + self.intrinsic_coef * rewards_int[t]
                for t in range(T)
            ]

            returns = []
            G = 0.0
            for r in reversed(combined):
                G = r + self.gamma * G
                returns.insert(0, G)

            returns_t = torch.FloatTensor(returns).to(self.device)
            values_t = torch.stack(values).squeeze()
            log_probs_t = torch.stack(log_probs).squeeze()
            entropies_t = torch.stack(entropies).squeeze()

            advantages = returns_t - values_t.detach()
            policy_loss = -(log_probs_t * advantages).mean()
            value_loss = F.mse_loss(values_t, returns_t)
            entropy_loss = -entropies_t.mean()

            total_loss = (
                policy_loss
                + self.vf_coef * value_loss
                + self.ent_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.policy_head.parameters()) +
                list(self.value_head.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()
