from typing import Any
import numpy as np 
import torch 
import torch.nn as nn
from torch.nn.functional import relu, softmax

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dim, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    
    def forward(self, state):
        x = relu(self.fc1(state))
        x = relu(self.fc2(x))
        x = self.fc3(x) # logits 

        return x 


class PolicyGradientAgent():
    def __init__(self, lr, input_dims, gamma=0.99, n_actions = 4):
        self.gamma = gamma 
        self.lr = lr 
        self.reward_memory = []
        self.action_memory = []
        self.episodes_trained = 0

        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

    def choose_action(self,observation):
        state = torch.Tensor(observation).to(self.policy.device)       
        logits =  self.policy(state)
        probabilities = softmax(logits, dim=0)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()

        # G_t = sum from k=0 to k=T {gamma**k *R_t+k+1}

        G = np.zeros_like(self.reward_memory)

        # for t in range(len(self.reward_memory)):
        #     G_sum = 0 
        #     discount = 1
        #     for k in range(t, len(self.reward_memory)):
        #         G_sum += self.reward_memory[k] * discount 
        #         discount *= self.gamma 
        #     G[t] = G_sum 

        G[0] = np.sum([self.gamma**k * self.reward_memory[k] for k in range(len(self.reward_memory))])

        for t in range(1,len(self.reward_memory)):
            G[t] = (G[t-1] - self.reward_memory[t-1])/self.gamma
        
        G = torch.Tensor(G).to(self.policy.device)

        loss = 0 
        for g, logprob in zip(G,self.action_memory):
            loss += -g * logprob
        
        loss.backward()


        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
        self.episodes_trained += 1















    
