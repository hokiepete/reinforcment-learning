import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_eval_det import print_policy, print_values

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def play_game(grid, policy):

    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    a = np.random.choice(ALL_POSSIBLE_ACTIONS)

    states_actions_rewards = [(s, a, 0)]
    ct = 0
    seen = set()
    seen.add(grid.current_state())
    while True:
        ct += 1
        # print(a, flush=True)
        r = grid.move(a)
        s = grid.current_state()
        if s in seen:
            states_actions_rewards.append((s, None, -10/ct))
            break
        elif grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = policy[s]
            states_actions_rewards.append((s, a, r))
        seen.add(s)
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in states_actions_rewards[::-1]:
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))

        G = r + GAMMA * G
    return states_actions_returns[::-1]

def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

if __name__ == '__main__':

    grid = negative_grid(step_cost=-0.15)
    #grid = standard_grid()
    print("rewards:")
    print_values(grid.rewards, grid)

    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    Q = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0
                returns[(s, a)] = []

    deltas = []
    for t in range(2000):
        if t % 1000 == 0:
            print(t, flush=True)
        biggest_change = 0
        states_actions_returns = play_game(grid, policy)
        seen_state_action_pairs = set()
        for s, a, G in states_actions_returns:
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(
                    biggest_change, abs(old_q - Q[s][a])
                )
                seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)

        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]

    print('policy:')
    print_policy(policy, grid)

    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print_values(V, grid)
