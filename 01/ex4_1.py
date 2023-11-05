import gym

env = gym.make('CartPole-v1')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)