import gym

env = gym.make('CartPole-v1', render_mode="human")
env.reset()
for _ in range(1000):
    env.render()
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample()) # take a random action
env.close()