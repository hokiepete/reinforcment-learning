import matplotlib.pyplot as plt
import numpy as np

NUM_TRAILS = int(1e+5)
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

def ucb(band, n):
    m = -999
    m_i = -1
    for i, b in enumerate(band):
        val = b.p_est + np.sqrt(2*np.log(n)/b.N)
        if  val > m:
            m = val
            m_i = i
    return m_i


def experiment(bandit_probs):
    banditos = [Bandit(x) for x in bandit_probs]
    N = 0
    for bandito in banditos:
        bandito.update(bandito.pull())
        N+=1

    for _ in range(NUM_TRAILS):
        
        j = ucb(banditos, N)

        banditos[j].update(banditos[j].pull())
        N+=1

    for bandito in banditos:
        print(bandito.N, bandito.p, bandito.p_est)

if __name__ == '__main__':
    bandits = [0.2, 0.5, 0.75]
    experiment(bandits)