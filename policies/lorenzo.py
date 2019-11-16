import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_layer=(256, 256)):
        super(Actor, self).__init__()
        self.is_recurrent = False
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.l1 = nn.Sequential(
                nn.Linear(state_dim, hidden_layer[0]),
                nn.ReLU()
              )
        self.l2 = nn.Sequential(
                nn.Linear(hidden_layer[0], hidden_layer[1]),
                nn.ReLU()
              )
        self.l3 = nn.Sequential(
                nn.Linear(hidden_layer[1], action_dim),
                nn.Tanh()
              )

        self.max_action = max_action #torch.FloatTensor(max_action)


    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.max_action * self.l3(x)

        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer=(256, 256)):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_layer[0]), nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(hidden_layer[0],        hidden_layer[1]), nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(hidden_layer[1],                      1),          )

        # Q2 architecture
        self.l4 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_layer[0]), nn.ReLU())
        self.l5 = nn.Sequential(nn.Linear(hidden_layer[0],        hidden_layer[1]), nn.ReLU())
        self.l6 = nn.Sequential(nn.Linear(hidden_layer[1],                      1),          )


    def forward(self, x, u):
        return self.Q1(x, u), self.Q2(x, u)


    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        # Q1 forward
        x1 = self.l1(xu)
        x1 = self.l2(x1)
        x1 = self.l3(x1)

        return x1


    def Q2(self, x, u):
        xu = torch.cat([x, u], 1)

        # Q2 forward
        x2 = self.l4(xu)
        x2 = self.l5(x2)
        x2 = self.l6(x2)

        return x2
