import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.replay_buffer import ReplayBuffer
from utils.config_loader import load_config


class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDQNAgent:
    def __init__(self, state_dim, action_dim, cfg_override=None, action_costs=None):
        self.cfg = cfg_override if cfg_override else load_config()


        self.lr = self.cfg['DDQN']['LEARNING_RATE_DDQN']
        self.gamma = self.cfg['DDQN']['gamma']
        self.epsilon = self.cfg['DDQN']['epsilon_start']
        self.epsilon_min = self.cfg['DDQN']['epsilon_min']
        self.epsilon_decay = self.cfg['DDQN']['epsilon_decay']
        self.batch_size = self.cfg['DDQN']['BATCH_SIZE']
        self.buffer_size = self.cfg['DDQN']['BUFFER_SIZE']
        self.training_interval = self.cfg['DDQN']['TRAINING_INTERVAL']
        self.tau = self.cfg['DDQN']['TAU']


        self.state_dim = state_dim
        self.action_dim = action_dim


        tie_cfg = self.cfg.get("DDQN", {}).get("tie_break", {}) or {}
        self.tie_break_enabled = bool(tie_cfg.get("enabled", False))
        self.tie_break_q_tolerance = float(tie_cfg.get("q_tolerance", 1.0e-6))
        if self.tie_break_q_tolerance < 0.0:
            raise ValueError("DDQN.tie_break.q_tolerance must be non-negative.")
        if action_costs is None:
            self.action_costs = None
            if self.tie_break_enabled:
                raise ValueError(
                    "Tie-breaking is enabled but action_costs were not supplied to DDQNAgent."
                )
        else:
            costs = torch.as_tensor(action_costs, dtype=torch.float32).flatten()
            if costs.numel() != self.action_dim:
                raise ValueError(
                    f"action_costs must contain {self.action_dim} values; "
                    f"found {costs.numel()}."
                )
            if not torch.isfinite(costs).all() or torch.any(costs < 0):
                raise ValueError("action_costs must be finite and non-negative.")
            self.action_costs = costs


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.policy_net = DDQN(state_dim, action_dim).to(self.device)
        self.target_net = DDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())


        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)


        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)


        self.steps_done = 0


        self.training_history = {'loss': [], 'episode_rewards': [], 'epsilon': [], 'q_values': []}

    def _greedy_indices_from_q(self, q_values):
        """Return greedy action indices with deterministic low-cost tie-breaking.

        ``q_values`` may be one-dimensional or batched. When tie-breaking is
        disabled, this is identical to ``argmax``. When enabled, all actions
        within the configured absolute tolerance of the row maximum are
        considered equivalent and the action with the lowest supplied cost is
        selected; the lowest action index resolves any remaining exact tie.
        """
        squeeze = q_values.ndim == 1
        values = q_values.unsqueeze(0) if squeeze else q_values
        if not self.tie_break_enabled:
            result = values.argmax(dim=1)
            return result[0] if squeeze else result

        max_values = values.max(dim=1, keepdim=True).values
        candidates = values >= (max_values - self.tie_break_q_tolerance)
        costs = self.action_costs.to(values.device).unsqueeze(0).expand_as(values)
        masked_costs = torch.where(
            candidates, costs, torch.full_like(costs, float("inf"))
        )
        result = masked_costs.argmin(dim=1)
        return result[0] if squeeze else result

    def greedy_action(self, state):
        """Select the deterministic greedy action used in validation and test."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.policy_net(state_tensor).squeeze(0)
            return int(self._greedy_indices_from_q(q_values).item())

    def select_action(self, state, eval_mode=False, eval_epsilon=0.01):
        if eval_mode:

            if random.random() < eval_epsilon:
                return random.randint(0, self.action_dim - 1)

            return self.greedy_action(state)


        self.steps_done += 1

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        return self.greedy_action(state)

    def update_epsilon(self):

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_history['epsilon'].append(self.epsilon)

    def train(self):

        if len(self.memory) < self.batch_size * 5 or self.steps_done % self.training_interval != 0:
            return None


        states, actions, rewards, next_states, dones = self.memory.sample()


        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))


        with torch.no_grad():

            next_policy_values = self.policy_net(next_states)
            next_actions = self._greedy_indices_from_q(next_policy_values)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))


            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))


        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values.squeeze())


        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath, metadata=None):
        """Save the agent state and optional caller-supplied run metadata."""
        payload = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_history": self.training_history,
            "epsilon": float(self.epsilon),
            "steps_done": int(self.steps_done),
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "tie_break_enabled": bool(self.tie_break_enabled),
            "tie_break_q_tolerance": float(self.tie_break_q_tolerance),
        }
        if metadata:
            overlap = set(payload).intersection(metadata)
            if overlap:
                raise ValueError(
                    f"Metadata keys would overwrite core checkpoint fields: {sorted(overlap)}"
                )
            payload.update(metadata)
        torch.save(payload, filepath)

    def load(self, filepath, load_optimizer=True):
        """Load an agent checkpoint safely on the active CPU/GPU device."""
        try:
            checkpoint = torch.load(
                filepath, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(filepath, map_location=self.device)

        saved_state_dim = checkpoint.get("state_dim")
        saved_action_dim = checkpoint.get("action_dim")
        if saved_state_dim is not None and int(saved_state_dim) != self.state_dim:
            raise ValueError(
                f"Checkpoint state_dim={saved_state_dim} does not match agent state_dim={self.state_dim}."
            )
        if saved_action_dim is not None and int(saved_action_dim) != self.action_dim:
            raise ValueError(
                f"Checkpoint action_dim={saved_action_dim} does not match agent action_dim={self.action_dim}."
            )

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history = checkpoint.get("training_history", self.training_history)
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.steps_done = int(checkpoint.get("steps_done", self.steps_done))
        return checkpoint
