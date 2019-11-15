import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt

from policies.base import Net

class Actor(Net):
  def __init__(self):
    super(Actor, self).__init__()

  def forward(self):
    raise NotImplementedError

  def get_action(self):
    raise NotImplementedError

class Linear_Actor(Actor):
  def __init__(self, state_dim, action_dim, hidden_size=32):
    super(Linear_Actor, self).__init__()

    self.l1 = nn.Linear(state_dim, hidden_size)
    self.l2 = nn.Linear(hidden_size, action_dim)

    self.action_dim = action_dim

    for p in self.parameters():
      p.data = torch.zeros(p.shape)

  def forward(self, state):
    a = self.l1(state)
    a = self.l2(a)
    self.action = a
    return a

  def get_action(self):
    return self.action

class FF_Actor(Actor):
  def __init__(self, state_dim, action_dim, layers=(256, 256), env_name='NOT SET', nonlinearity=F.relu, normc_init=True):
    super(FF_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.Linear(state_dim, layers[0])]
    for i in range(len(layers)-1):
        self.actor_layers += [nn.Linear(layers[i], layers[i+1])]
    self.network_out = nn.Linear(layers[-1], action_dim)

    self.action = None
    self.action_dim = action_dim
    self.env_name = env_name
    self.nonlinearity = nonlinearity

    if normc_init:
      self.initialize_parameters()

  def forward(self, state):
    x = state
    for idx, layer in enumerate(self.actor_layers):
      x = self.nonlinearity(layer(x))

    self.action = torch.tanh(self.network_out(x))
    return self.action

  def get_action(self):
    return self.action

class LSTM_Actor(Actor):
  def __init__(self, input_dim, action_dim, layers=(128, 128), env_name='NOT SET', nonlinearity=torch.tanh, normc_init=True):
    super(LSTM_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.LSTMCell(input_dim, layers[0])]
    for i in range(hidden_layers-1):
        self.actor_layers += [nn.LSTMCell(layers[i], layers[i+1])]
    self.network_out = nn.Linear(layers[i-1], action_dim)

    self.action = None
    self.action_dim = action_dim
    self.init_hidden_state()
    self.env_name = env_name
    self.nonlinearity = nonlinearity
    
    self.is_recurrent = True

    if normc_init:
      self.initialize_parameters()

  def get_hidden_state(self):
    return self.hidden, self.cells

  def set_hidden_state(self, data):
    if len(data) != 2:
      print("Got invalid hidden state data.")
      exit(1)

    self.hidden, self.cells = data
    
  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
    self.cells  = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

  def forward(self, x):

    if len(x.size()) == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))
      action = []
      for t, x_t in enumerate(x):

        for idx, layer in enumerate(self.actor_layers):
          c, h = self.cells[idx], self.hidden[idx]
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]
        x_t = self.network_out(x_t)
        action.append(x_t)

      x = torch.stack([a.float() for a in action])
      self.action = x

    elif len(x.size()) == 2: # if we get a whole trajectory
      self.init_hidden_state()

      self.action = []
      for t, x_t in enumerate(x):
        x_t = x_t.view(1, -1)

        for idx, layer in enumerate(self.actor_layers):
          h, c = self.hidden[idx], self.cells[idx]
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]
        x_t = self.nonlinearity(self.network_out(x_t))
        self.action.append(x_t)

      x = torch.cat([a.float() for a in self.action])
      self.action = x


    elif len(x.size()) == 1: # if we get a single timestep
      x = x.view(1, -1)

      for idx, layer in enumerate(self.actor_layers):
        h, c = self.hidden[idx], self.cells[idx]
        self.hidden[idx], self.cells[idx] = layer(x, (h, c))
        x = self.hidden[idx]
      x = self.nonlinearity(self.network_out(x))[0]
      self.action = x

    else:
      print("Invalid input dimensions.")
      exit(1)

    return x
  
  def get_action(self):
    return self.action
