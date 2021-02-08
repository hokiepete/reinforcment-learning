import gym

env = gym.make('CartPole-v0')
base_state = env.reset()
box = env.observation_space
act = env.action_space

# observation, reward, done, info = env.step(action)

done = False
cnt = 0
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    cnt += 1
