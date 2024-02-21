from typing import Any
import numpy as np 
import torch as T
import torch.nn as nn
from torch.nn.functional import relu, softmax
import os

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dim, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
  
    def forward(self, state):
        x = relu(self.fc1(state))
        x = relu(self.fc2(x))
        x = self.fc3(x) # logits 
        return x 
    
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            self.load_state_dict(T.load(checkpoint_path))
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    
    def save_checkpoint(self, checkpoint_path):
        print(f"Saving checkpoint to {checkpoint_path}")
        T.save(self.state_dict(), checkpoint_path)


class PolicyGradientAgent():
    def __init__(self, lr, input_dims, gamma=0.99, n_actions = 4):
        self.gamma = gamma 
        self.lr = lr 
        self.reward_memory = []
        self.action_memory = []
        self.episodes_trained = 0

        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

    def choose_action(self,observation):
        state = T.Tensor(observation).to(self.policy.device)       
        logits =  self.policy(state)
        probabilities = softmax(logits, dim=0)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()


        G = np.zeros_like(self.reward_memory)

        G[0] = np.sum([self.gamma**k * self.reward_memory[k] for k in range(len(self.reward_memory))])

        for t in range(1,len(self.reward_memory)):
            G[t] = (G[t-1] - self.reward_memory[t-1])/self.gamma
        
        G = T.Tensor(G).to(self.policy.device)

        loss = 0 
        for g, logprob in zip(G,self.action_memory):
            loss += -g * logprob
        loss /= len(self.action_memory)
        # print(loss)
        loss.backward()


        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
        self.episodes_trained += 1
    
















    
