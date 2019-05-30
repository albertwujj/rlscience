import os
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F



class Policy():
    def __init__(self, nn_args, action_space, atoms, V_min, V_max, device, model):
        self.action_space = action_space
        self.atoms = atoms
        self.V_min = V_min
        self.V_max = V_max
        self.support = torch.linspace(V_min, V_max, atoms).to(device=device)  # Support (range) of z

        self.online_net = DQN(action_space, atoms, **nn_args).to(device=device)
        if model and os.path.isfile(model):
            # Always load tensors onto CPU by default, will shift to GPU if necessary
            self.online_net.load_state_dict(torch.load(model, map_location='cpu'))
        self.online_net.train()

        self.target_net = DQN(action_space, atoms, **nn_args).to(device=device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False


    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()
        # Acts based on single state (no batch)

    def act(self, state):
        with torch.no_grad():
            return (self.online_net(state) * self.support).sum(2).argmax(1).unsqueeze(1)

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return torch.from_numpy(np.random.randint(0, self.action_space, size=(state.shape[0], 1))) \
            if np.random.random() < epsilon else self.act(state)

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path):
        torch.save(self.online_net.state_dict(), os.path.join(path, 'model.pth'))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state) * self.support).sum(2).max(1)

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()



class DQN(nn.Module):
    def __init__(self, action_space, atoms, history_length, hidden_size, noisy_std):
        super().__init__()
        self.atoms = atoms
        self.action_space = action_space

        self.conv1 = nn.Conv2d(history_length, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc_h_v = NoisyLinear(3136, hidden_size, std_init=noisy_std)
        self.fc_h_a = NoisyLinear(3136, hidden_size, std_init=noisy_std)
        self.fc_z_v = NoisyLinear(hidden_size, self.atoms, std_init=noisy_std)
        self.fc_z_a = NoisyLinear(hidden_size, action_space * self.atoms, std_init=noisy_std)

    def forward(self, x, log=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)