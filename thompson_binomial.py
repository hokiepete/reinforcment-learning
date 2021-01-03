import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

NUM_TRAILS = int(1e+5)
EPS = 0.1

class Bandit:
    def __init__(self, p):
        self.p = p
        self.p_est = 0
        self.N = 0
        self.a = 1
        self.b = 1

    def sample(self):
        return np.random.beta(self.a,self.b)

    def pull(self):
        return np.random.random() < self.p

    def update(self, x):
        self.N+=1
        self.a += x
        self.b += 1 - x
        # self.p_est = beta(self.a,self.b+self.N).mean()


def experiment(bandit_probs):
    banditos = [Bandit(x) for x in bandit_probs]
    N = 0
    for _ in range(NUM_TRAILS):
        
        j = np.argmax([b.sample() for b in banditos])

        banditos[j].update(banditos[j].pull())
        N+=1

    for bandito in banditos:
        print(bandito.N, bandito.p, beta(bandito.a,bandito.b).mean())

if __name__ == '__main__':
    bandits = [0.2, 0.5, 0.75]
    experiment(bandits)