import numpy as np
import matplotlib.pyplot as plt 
import gymnasium as gym 
from agent import Agent

class CartPoleStateDigitizer():
    def __init__(self, bounds=(2.4, 4, 0.209, 4), n_bins=10):
        self.position_space = np.linspace(-1*bounds[0], bounds[0], n_bins)
        self.velocity_space = np.linspace(-1*bounds[1], bounds[1], n_bins)
        self.pole_angle_space = np.linspace(-1*bounds[2], bounds[2], n_bins)
        self.pole_velocity_space = np.linspace(-1*bounds[3], bounds[3], n_bins)
        self.states = self.get_state_space()

    
    def get_state_space(self):
        states = []
        for i in range(len(self.position_space)+1):
            for j in range(len(self.velocity_space)+1):
                for k in range(len(self.pole_angle_space)+1):
                    for l in range(len(self.pole_velocity_space)+1):
                        states.append((i,j,k,l))
        return states 
    

    def digitize(self, observation):
        x, x_dot, theta, theta_dot = observation
        cart_x = int(np.digitize(x, self.position_space))
        cart_xdot = int(np.digitize(x_dot, self.velocity_space))
        cart_theta = int(np.digitize(theta, self.pole_angle_space))
        cart_thetadot = int(np.digitize(theta_dot, self.pole_velocity_space))

        return (cart_x, cart_xdot, cart_theta, cart_thetadot)
    
def plot_learning_curve(scores, x):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0,i-100):(i+1)])
    plt.plot(x,running_avg)
    plt.title("Running average of previous 100 scores")
    plt.show()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 50000
    eps_dec = 2/n_games
    digitizer = CartPoleStateDigitizer()
    agent = Agent(lr=0.01, gamma=0.99, n_actions=2, eps_start=1.0, 
                  eps_end=0.01, eps_dec=eps_dec, state_space=digitizer.states)
    scores = []

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        state = digitizer.digitize(observation)
        score = 0
        while not done:
            action = agent.choose_action(state)
            (observation_, reward, done, info, _) = env.step(action)
            state_ = digitizer.digitize(observation_)
            agent.learn(state, action, reward, state_)
            state = state_
            score +=reward 

        if i % 1000 == 0:
            print('episode', i, 'score %.1f' % score, 'epsilon %.2f' %agent.epsilon)
        
        agent.decrement_epsilon()
        scores.append(score)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(scores, x)
