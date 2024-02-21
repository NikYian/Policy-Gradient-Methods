import gymnasium as gym 
import imageio
from Agent import PolicyGradientAgent
import matplotlib.pyplot as plt 
import numpy as np

def plot_learning_curve(scores, N=100, figure_file='plot'):
    running_avg = np.convolve(scores, np.ones(N)/N, mode='valid') 
    plt.plot(running_avg)
    plt.title('Running Average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    n_games = 5000
    # video_filename = 'training_video.mp4'
    # video_writer = imageio.get_writer(video_filename, fps=30)

    obs = env.reset()[0]
    
    agent = PolicyGradientAgent(lr=5e-4, input_dims=obs.shape)
    score_ls = []

    model_name = 'REINFORCE_' 

    for i in range(n_games):
        obs = env.reset()[0]

        score = 0 
        done = False 
        trunc = False 
        while (not done) and (not trunc):
            action = agent.choose_action(obs)
            obs, reward, done, trunc, info = env.step(action)
            score +=reward 
            agent.store_reward(reward)
            # img = env.render()

            # img = env.render()
            # video_writer.append_data(img)

        agent.learn()
        score_ls.append(score)
        avg_score = np.mean(score_ls[-100:])
        print('episode ', i, 'score %.1f' %score, 
                'average score %.1f' %avg_score)
    
    # video_writer.close()
    env.close()

    plt.plot(score_ls, label='Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Performance')
    plt.legend()
    plt.show()

    # Save score_ls array
    np.save(model_name + 'scores.npy', score_ls)

    # # Save agent's policy checkpoint
    # agent.policy.save_checkpoint(model_name + 'trained_agent.pth')

        #test
    test_scores = []
    for _ in range(100):
        obs = env.reset()[0]
        score = 0
        done = False
        trunc = False
        while (not done) and (not trunc):
            action = agent.choose_action(obs)
            obs, reward, done, trunc, info = env.step(action)
            score +=reward
        test_scores.append(score)

    mean_score  = np.mean(test_scores)
    std_score = np.std(test_scores)
    file_path = 'scores.txt'

    # Append mean and standard deviation scores to the file
    with open(file_path, 'a') as file:
        file.write(f"Mean Score of 100 episodes: {mean_score} ")
        file.write(f"std: {std_score}\n\n")

    print(f"Scores appended to {file_path}")