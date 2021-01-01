import matplotlib.pyplot as plt
import numpy as np

NUM_TRAILS = int(1e+4)
EPS = 0.1

class Bandit:
    def __init__(self, p):
        self.p = p
        self.p_est = 0
        self.N = 0

    def pull(self):
        return np.random.random() < self.p

    def update(self, x):
        self.N+=1
        self.p_est = self.p_est + (1/self.N) * (x - self.p_est)

def amax(band):
    m = -999
    m_i = -1
    for i, b in enumerate(band):
        if b.p_est > m:
            m = b.p_est
            m_i = i
    return m_i


def experiment(bandit_probs):
    banditos = [Bandit(x) for x in bandit_probs]
    opt_i = 0
    for _ in range(NUM_TRAILS):
        if np.random.random() < EPS:
            j = np.random.choice([i for i in range(len(banditos)) if i != opt_i])
        else:
            j = opt_i

        banditos[j].update(banditos[j].pull())

        opt_i = amax(banditos)

    for bandito in banditos:
        print(bandito.N, bandito.p, bandito.p_est)

if __name__ == '__main__':
    bandits = [0.2, 0.5, 0.75]
    experiment(bandits)