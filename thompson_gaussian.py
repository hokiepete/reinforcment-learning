import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

NUM_TRAILS = int(1e+6)
EPS = 0.1
T = 1

class Bandit:
    def __init__(self, p):
        self.p = p
        self.p_est = 0
        self.N = 0
        self.l = 1
        self.t = 1
        self.sum_x = 0

    def sample(self):
        return np.random.randn() / np.sqrt(self.l) + self.p_est

    def pull(self):
        return np.random.randn() / np.sqrt(self.t) + self.p

    def update(self, x):
        self.N+=1
        self.l += self.t
        self.sum_x += x
        self.p_est = self.t*self.sum_x / self.l


def experiment(bandit_probs):
    banditos = [Bandit(x) for x in bandit_probs]
    N = 0
    for _ in range(NUM_TRAILS):
        
        j = np.argmax([b.sample() for b in banditos])

        banditos[j].update(banditos[j].pull())
        N+=1

    for bandito in banditos:
        print(bandito.N, bandito.p, norm(loc=bandito.p_est,scale=1/bandito.l).mean())

if __name__ == '__main__':
    bandits = [0.2, 0.5, 0.75]
    experiment(bandits)