import numpy as np
import torch as T
import torch.nn as nn 
import torch.nn.functional as F
    
class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=128, fc2_dims=128):
        super(ActorNetwork,self).__init__()
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

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=64, fc2_dims=128, fc3_dims=68):
        super(CriticNetwork,self).__init__()
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
    
class Agent():
    def __init__(self, a_lr, c_lr,  input_dims, n_actions, gamma=0.99):
        self.gamma = gamma 
        self.actor = ActorNetwork(a_lr, input_dims, n_actions)
        self.critic = CriticNetwork(c_lr, input_dims)
        self.log_prob = None

    def choose_action(self, observation):
        state = T.tensor([observation]).to(self.actor.device)
        logits = self.actor(state)
        probabilities = F.softmax(logits, dim=1) 
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action) 
        self.log_prob = log_prob

        return action.item()
    
    def learn(self, state, reward, state_, done):
        self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        device = self.actor.device
        state = T.tensor(state).to(device)
        reward = T.tensor(reward).to(device)
        state_ = T.tensor(state_).to(device) 

        critic_value = self.critic(state)
        critic_value_ = self.critic(state_)

        delta = reward + self.gamma*critic_value_ *(1-int(done)) -critic_value 

        actor_loss = -self.log_prob*delta.detach()
        actor_loss.backward() 
        self.actor.optimizer.step()

        critic_loss = delta**2 
        critic_loss.backward()
        self.critic.optimizer.step()

