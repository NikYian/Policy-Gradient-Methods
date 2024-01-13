import gymnasium as gym
import imageio

if __name__ == '__main__':
    env = gym.make('LunarLander-v2',  render_mode="rgb_array")

    n_games = 100
    video_filename = 'training_video.mp4'
    video_writer = imageio.get_writer(video_filename, fps=30)

    for i in range(n_games):
        obs = env.reset()

        score = 0 
        done = False 
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, trunc, info = env.step(action)
            score +=reward 
            # img = env.render()

            # img = env.render()
            # video_writer.append_data(img)


        print('episode', i, 'score %.1f' %score)
    
    # video_writer.close()
    env.close()
