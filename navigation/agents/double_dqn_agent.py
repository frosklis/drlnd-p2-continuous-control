import torch
import torch.nn.functional as F
from .dqn_agent import DQN_Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Double_DQN_Agent(DQN_Agent):
    """Interacts with and learns from the environment."""
    def __init__(self, environment,
                 description='Double DQN Agent',
                 **kwargs):
        super().__init__(environment, description=description, **kwargs)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get predicted Q values (for next states) from target model
        # using the actions that the online model would choose
        argmax_next = self.qnetwork_local(
            next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(
            next_states).detach().gather(1, argmax_next)
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
