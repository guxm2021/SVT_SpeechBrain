import torch
import torch.nn as nn


class DiscNet(nn.Module):
    """
    Discriminator doing binary classification: source v.s. target
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    

class ClassDiscNet(nn.Module):
    """
    Discriminator doing binary classification: source v.s. target
    """
    def __init__(self, input_dim, domain_dim, hidden_dim, output_dim, cond=False):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_t = nn.Linear(domain_dim, hidden_dim)
        if cond:
            self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.cond = cond
    
    def forward(self, x, t=None):
        x = torch.relu(self.fc1(x))
        if self.cond:
            t = torch.relu(self.fc_t(t))
            x = torch.cat([x, t], dim=-1)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x