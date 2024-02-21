from typing import Any
import numpy as np 
import torch as T
import torch.nn as nn
from torch.nn.functional import relu, softmax, softplus
import torch.nn.utils as torch_utils

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dim, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, n_actions)
        self.sigma = nn.Linear(128, n_actions)
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr,weight_decay=1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = relu(self.fc1(state))
        x = relu(self.fc2(x))
        mu = self.mu(x)
        sigma = softplus(self.sigma(x))
        return mu, sigma
    
class PolicyGradientAgent():
    def __init__(self, lr, input_dims, gamma=0.99, n_actions = 2):
        self.gamma = gamma 
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyNetwork(lr, input_dims, n_actions)
    
    def choose_action(self, observation):
        state = T.Tensor(observation).to(self.policy.device)
        mu, sigma = self.policy(state)
        cov_matrix = T.diag_embed(sigma**2) 
        dist = T.distributions.MultivariateNormal(mu, cov_matrix)
        action = dist.sample()
        # dist = T.distributions.MultivariateNormal(mu.detach(), cov_matrix.detach())
        log_probs = dist.log_prob(action)
        self.action_memory.append(log_probs)

        return action.cpu().detach().numpy()

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()

        G = np.zeros_like(self.reward_memory)
        G[0] = np.sum([self.gamma**k * self.reward_memory[k] for k in range(len(self.reward_memory))])

        for t in range(1,len(self.reward_memory)):
            G[t] = (G[t-1] - self.reward_memory[t-1])/self.gamma
        # breakpoint()
        G = T.Tensor(G).to(self.policy.device)
        loss = 0 
        for g, logprob in zip(G,self.action_memory):
            loss += -g * logprob
        loss.backward()

        self.policy.optimizer.step()
        self.action_memory = []
        self.reward_memory = []

    
















    
