rm import random
from collections import deque, namedtuple
from typing import Tuple

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from bananas.model import QNetwork
from bananas.problem import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN_Agent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, environment,
                 learning_rate: float = 5e-4,
                 gamma: float = 0.99,
                 buffer_size=int(1e5),
                 batch_size=64,
                 update_every=4,
                 tau=1e-3,
                 alpha: float = 0,
                 memory_e: float = 0.0001,
                 memory_rank_based: bool = False,
                 description='DQN Agent',
                 partial_network=QNetwork,
                 **kwargs):
        super().__init__(environment, description=description, **kwargs)
        self.tau = tau

        # Q-Network
        self.qnetwork_local = partial_network(
            self.state_size, self.action_size, seed=self.seed).to(device)
        self.qnetwork_target = partial_network(
            self.state_size, self.action_size, seed=self.seed).to(device)

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = ReplayBuffer(self.action_size, buffer_size, batch_size, self.seed)
        self.t_step = 0
        self.update_every = update_every

        # Params for the Memory Buffer
        self.alpha = alpha
        self.memory_e = memory_e
        self.memory_rank_based = memory_rank_based

    def learn_from_step(self, state, action, reward, next_state, done):

        with torch.no_grad():
            next_state_mem = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
            state_mem = torch.from_numpy(state).float().unsqueeze(0).to(device)

            # Calculate the error (priority) and save experience in replay memory
            priority = np.abs(reward + self.gamma * self.qnetwork_local(next_state_mem).cpu().data.numpy().max() -
                              self.qnetwork_local(state_mem).cpu().data.numpy()[0, action])

        self.memory.add(state, action, reward, next_state, priority, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(alpha=self.alpha, e=self.memory_e, rank_based=self.memory_rank_based)
                self.learn(experiences, self.gamma)

    def learn(self, experiences: Tuple[torch.Tensor], gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    @property
    def model_class(self):
        return type(self.qnetwork_local)

    def eval(self, state):
        torch_state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(torch_state)
        self.qnetwork_local.train()
        return action_values.cpu().data.numpy()

    def log_initial_setup(self):
        super().log_initial_setup()
        has_memory = self.buffer_size > 1
        mlflow.log_param('memory', has_memory)
        if has_memory:
            mlflow.log_param('experience_replay', 'random' if self.alpha == 0 else 'prioritized')
            mlflow.log_param('priority', 'rank based' if self.memory_rank_based else 'error based')

        mlflow.log_params({
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'tau': self.tau,
            'learning_rate': self.learning_rate,
            'update_every': self.update_every,
            'memory_a': self.alpha,
            'memory_e': self.memory_e
        })
        mlflow.set_tag('qnetwork', str(self.qnetwork_local))

    def _save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename)

    @classmethod
    def load(cls, model_class, model_file, environment):
        instance = cls(environment, partial_network=model_class)
        instance.qnetwork_local.load_state_dict(torch.load(model_file))
        instance.qnetwork_target.load_state_dict(torch.load(model_file))
        return instance


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
            "state", "action", "reward", "next_state", "priority", "done"])
        self.seed = random.seed(seed)
        self._last_returned_indices = None

    def add(self, state, action, reward, next_state, priority, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, priority, done)
        self.memory.append(e)

    def sample(self, alpha: float = 0, e: float = 0.0001, rank_based: bool = False, return_probabilites: bool = False):
        """Randomly sample a batch of experiences from memory.

        The probability of an experience getting sampled depends on the priority, or how close the reward is to the
         expected reward. The lower the error, the lower the chance of getting picked.
        The default behavior is, however, completeley random selection.

        See https://arxiv.org/abs/1511.05952 for full explanation. The implementation is not so neat.

        :param alpha: priorization exponent,  use a=0 for uniform sampling, a=1 for probability based sampling
        :param e: small value to prevent edge case
        :param rank_based: compute probabilities based on rankings rather than absolute error magnitudes, theoretically (and intuitively) more robust
        :param return_probabilites: whether to return probabilities or not
        :return: tuple of selected experiences, the structure is different depending on the value of retur_probabilities:
        (states, actions, rewards, next_states, dones, probabilities) if True
        (states, actions, rewards, next_states, dones) if False

        """
        if alpha == 0:  # Uniform sampling
            experiences = random.sample(self.memory, k=self.batch_size)
            probabilities = None
        else:
            if rank_based:
                priorities = (1 / (1 + torch.from_numpy(
                    np.vstack([e.priority for e in self.memory if e is not None])).to(device)[:,
                                       0].argsort().float())) ** alpha
            else:
                priorities = (e + torch.from_numpy(
                    np.vstack([e.priority for e in self.memory if e is not None])).float().to(device)[:, 0]) ** alpha
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(np.arange(0, len(self.memory)), size=self.batch_size, replace=False,
                                       p=probabilities.numpy())
            self._last_returned_indices = indices
            experiences = [self.memory[index] for index in indices]

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        if return_probabilites:
            return (states, actions, rewards, next_states, dones, probabilities)
        else:
            return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def update(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.memory[index]["priority"] = priority
