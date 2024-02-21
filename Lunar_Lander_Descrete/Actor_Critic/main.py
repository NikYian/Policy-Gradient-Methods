import gymnasium as gym 
import numpy as np
from Agent import Agent 
import matplotlib.pyplot as plt
import imageio
import os 

if __name__ == '__main__':
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    agent = Agent(gamma=.99, a_lr=5e-5, c_lr=5e-5, input_dims=[8], n_actions = 4)
    n_games = 5000
    
    model_name = 'ACTOR_CRITIC_'

    score_ls = []



    for i in range(n_games):
        done = False 
        obs = env.reset()[0]
        score = 0 
        trunc = False 

        if i in [1, 1000, 2000, 3000, 4000, 4999]:
            video_filename = 'A-C_episode:' + str(i) + '.mp4'
            video_writer = imageio.get_writer(video_filename, fps=30)

        while (not done) and (not trunc):
            action = agent.choose_action(obs)
            obs_, reward, done, trunc, info = env.step(action)
            score += reward 
            agent.learn(obs, reward, obs_, done)
            obs = obs_

            if i in [1, 1000, 2000, 3000, 4000, 4999]:
                img = env.render()
                video_writer.append_data(img)

        if i in [1, 1000, 2000, 3000, 4000, 4999]:
            video_writer.close()
            os.rename(video_filename, 'score:' + str(score) + video_filename)


        score_ls.append(score) 
        avg_score = np.mean(score_ls[-100:]) 
        print('episode ', i, 'score %.1f' %score, 'average score %.1f' %avg_score)
        
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