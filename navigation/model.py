import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=None, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.seed = seed
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=None,
                 fc1_units=64, fc2_units=64,
                 value_fc1_units=32,
                 advantage_fc1_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.seed = seed
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # Dueling architecture
        # value
        self.value_fc1 = nn.Linear(fc2_units, value_fc1_units)
        self.value_fc2 = nn.Linear(value_fc1_units, 1)

        # advantage
        self.advantage_fc1 = nn.Linear(fc2_units, advantage_fc1_units)
        self.advantage_fc2 = nn.Linear(advantage_fc1_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        action = F.relu(self.advantage_fc1(x))
        action = self.advantage_fc2(action)

        return value + action - action.mean(dim=1, keepdim=True)
