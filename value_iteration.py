
import numpy as np
from windy_grid_world import windy_grid, ACTION_SPACE
from iterative_policy_eval_prob import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9

def get_transition_probs_and_rewards(grid):

    transition_probs = {}
    rewards = {}

    for (s, a), v in grid.probs.items():
        for s2, p in v.items():
            transition_probs[(s, a, s2)] = p
            rewards[(s, a, s2)] = grid.rewards.get(s2, 0)

    return transition_probs, rewards



if __name__ == '__main__':

    grid = windy_grid()
    transition_probs, rewards = get_transition_probs_and_rewards(grid)

    print('rewards:')
    print_values(grid.rewards, grid)

    V = {}
    for s in grid.all_states():
        V[s] = 0

    it = 0
    while True:
        biggest_change = 0
        for s in grid.all_states():
            if grid.is_terminal(s):
                continue
            old_v = V[s]
            new_v = float('-inf')
            for a in ACTION_SPACE:
                v = 0
                for s2 in grid.all_states():
                    
                    r = rewards.get((s, a, s2), 0)
                    v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])
            
                if v > new_v:
                    new_v = v
            
            V[s] = new_v
            biggest_change = max(
                biggest_change, abs(old_v - new_v)
            )
        it+=1

        if biggest_change < SMALL_ENOUGH:
            break

    policy = {}
    for s in grid.actions.keys():
        best_a = None
        best_value = float('-inf')

        for a in ACTION_SPACE:
            v = 0
            for s2 in grid.all_states():
                r = rewards.get((s, a, s2), 0)
                v += transition_probs.get(
                    (s, a, s2), 0
                    ) * (r + GAMMA * V[s2])
            
            if v > best_value:
                best_value = v
                best_a = a
        policy[s] = best_a
    print('values:')
    print_values(V, grid)

    print('policy:')
    print_policy(policy, grid)