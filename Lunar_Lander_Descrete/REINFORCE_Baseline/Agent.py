from typing import Any
import numpy as np 
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=128, fc2_dims=128):
        super(PolicyNetwork,self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions) 

        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        return pi

class ValueNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=128, fc2_dims=128, fc3_dims=128):
        super(ValueNetwork,self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.v = nn.Linear(fc3_dims, 1) 

        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        v = self.v(x)
        return v


class PolicyGradientAgent():
    def __init__(self, p_lr,v_lr,  input_dims, gamma=0.99, n_actions = 4):
        self.gamma = gamma 

        self.reward_memory = []
        self.action_memory = []
        self.value_estimate_memory = []
        self.episodes_trained = 0

        self.policy = PolicyNetwork(p_lr, input_dims, n_actions)
        self.value_estimator = ValueNetwork(v_lr, input_dims)

    def choose_action(self,observation):
        state = T.Tensor(observation).to(self.policy.device)       
        logits =  self.policy(state)
        value_estimate = self.value_estimator(state)
        probabilities = F.softmax(logits, dim=0)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        self.value_estimate_memory.append(value_estimate)

        return action.item()

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        self.value_estimator.optimizer.zero_grad()


        G = np.zeros_like(self.reward_memory)

        G[0] = np.sum([self.gamma**k * self.reward_memory[k] for k in range(len(self.reward_memory))])

        for t in range(1,len(self.reward_memory)):
            G[t] = (G[t-1] - self.reward_memory[t-1])/self.gamma

        
        G = T.Tensor(G).to(self.policy.device)

        value_estimate_memory = T.stack(self.value_estimate_memory)
        action_memory = T.stack(self.action_memory)

        deltas = G - value_estimate_memory
        deltas_nograd = deltas.detach()

        actor_loss = T.sum( - deltas_nograd * action_memory) / len(deltas)
        actor_loss.backward()
        self.policy.optimizer.step()


        value_estimator_loss = T.sum(deltas**2)/ len(deltas)
        value_estimator_loss.backward()
        self.value_estimator.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
        self.value_estimate_memory = []
        self.episodes_trained += 1















    
