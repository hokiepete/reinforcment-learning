

class MyTradingSimulationEnv:
    def __init__(self):
        pass
    
    def reset(self):
        pass

    def step(self, action):
        pass

def get_actions(state):
    pass


env = MyTradingSimulationEnv()
done = False
state = env.reset()

while not done:
    action = get_actions(state)

    next_state, reward, done, info = env.step(action)

    state = next_state