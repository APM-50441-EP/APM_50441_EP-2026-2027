"""Utilities for Lab 6 (APM_50441_EP): Cliff Walking and the random walk of the lecture.

Only the environments, their tabular models and drawing helpers are here: the algorithms are written in the lab.

Cliff Walking (Sutton and Barto, Example 6.6) is gymnasium's `CliffWalking-v1`: a 4 x 12 grid, start S in the
bottom-left corner, goal G in the bottom-right corner, the cliff in between. Each step costs -1; stepping into the
cliff costs -100 and sends the agent back to S; reaching G ends the episode. The lecture also uses a *slippery*
version: with probability `slip`, the chosen action is replaced by a uniformly random one (`SlipActionWrapper`).
(gymnasium's own `CliffWalkingSlippery-v1` is a different model, with perpendicular slips.)

Actions: 0 = up, 1 = right, 2 = down, 3 = left. States: s = 12 * row + column (row 0 at the top).

Run this file to check the installation: python RL.py
"""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

NROW, NCOL = 4, 12
NS, NA = NROW * NCOL, 4
UP, RIGHT, DOWN, LEFT = range(4)
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
START, GOAL = (NROW - 1) * NCOL, NROW * NCOL - 1
CLIFF = [(NROW - 1) * NCOL + c for c in range(1, NCOL - 1)]
VALID = np.array([s not in CLIFF for s in range(NS)])  # the cliff cells are never occupied


class SlipActionWrapper(gym.ActionWrapper):
    """With probability `slip`, replace the action by a uniformly random one (the intended one included)."""

    def __init__(self, env, slip=0.1):
        super().__init__(env)
        self.slip = slip

    def action(self, action):
        if self.slip > 0 and self.np_random.random() < self.slip:
            return int(self.np_random.integers(self.action_space.n))
        return action


def make_env(slip=0.0, max_steps=None, render_mode=None):
    """Cliff Walking, optionally slippery, optionally with a maximum number of steps per episode."""
    env = gym.make('CliffWalking-v1', render_mode=render_mode, max_episode_steps=max_steps)
    return SlipActionWrapper(env, slip) if slip > 0 else env


def reset_to(env, state, seed=None):
    """Reset the environment, then move the agent to `state` (to start episodes anywhere)."""
    env.reset(seed=seed)
    env.unwrapped.s = int(state)
    return int(state)


def tabular_model(slip=0.0):
    """The model of the (slippery) Cliff Walking: P[s, a, s'] and the expected reward R[s, a].

    The goal and the cliff cells are absorbing with zero reward, so that the value of a policy is the expected sum
    of the rewards until the goal is reached.
    """
    base = gym.make('CliffWalking-v1').unwrapped.P
    P, R = np.zeros((NS, NA, NS)), np.zeros((NS, NA))
    for s in range(NS):
        for a in range(NA):
            if s == GOAL or s in CLIFF:
                P[s, a, s] = 1.0
                continue
            for b in range(NA):
                p = (1 - slip) * (a == b) + slip / NA
                ((_, s2, r, _),) = base[s][b]
                P[s, a, s2] += p
                R[s, a] += p * r
    return P, R


def greedy_path(actions, max_len=100):
    """States visited from S when following a deterministic policy (one action per state) without slips."""
    P, _ = tabular_model(0.0)
    path, s = [START], START
    while s != GOAL and len(path) < max_len:
        s = int(P[s, actions[s]].argmax())
        path.append(s)
    return path


SAFE_PATH = [START] + [r * NCOL for r in (2, 1, 0)] + list(range(1, NCOL)) + [r * NCOL + NCOL - 1 for r in (1, 2, 3)]
OPTIMAL_PATH = [START] + [(NROW - 2) * NCOL + c for c in range(NCOL)] + [GOAL]


def _xy(s):
    r, c = divmod(s, NCOL)
    return c, r


def draw_grid(ax, values=None, actions=None, paths=(), title=None, cmap='Blues_r'):
    """Draw the grid with the cliff, an optional value heatmap, optional greedy arrows and paths.

    values: array of NS values; actions: array of NS actions; paths: list of (states, color, label).
    """
    if values is not None:
        img = np.ma.masked_array(np.asarray(values, dtype=float).reshape(NROW, NCOL), ~VALID.reshape(NROW, NCOL))
        im = ax.imshow(img, cmap=cmap, origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    else:
        ax.imshow(np.ones((NROW, NCOL)), cmap=ListedColormap(['white']), origin='upper')
    for s in CLIFF:
        x, y = _xy(s)
        ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, color='0.35'))
    ax.text(5.5, NROW - 1, 'Cliff (-100)', color='white', ha='center', va='center')
    for s, letter in ((START, 'S'), (GOAL, 'G')):
        ax.text(*_xy(s), letter, ha='center', va='center', fontweight='bold', color='darkorange', zorder=5)
    if actions is not None:
        for s in range(NS):
            if VALID[s] and s not in (START, GOAL):
                x, y = _xy(s)
                dy, dx = MOVES[actions[s]]
                ax.arrow(x - .2 * dx, y - .2 * dy, .25 * dx, .25 * dy, head_width=.15, head_length=.12, color='0.2')
    for path, color, label in paths:
        xy = np.array([_xy(s) for s in path], dtype=float)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=3, label=label, zorder=4)
    if paths:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=len(paths), frameon=False)
    ax.set_xticks(np.arange(-.5, NCOL), minor=True)
    ax.set_yticks(np.arange(-.5, NROW), minor=True)
    ax.grid(which='minor', color='0.6', lw=.8)
    ax.tick_params(which='both', length=0, labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title)


def random_walk_episode(rng, n=5):
    """One episode of the random walk of Sutton and Barto (Example 6.2).

    States 0..n-1 (A..E for n = 5), start in the middle, move left or right with probability 1/2; the episode ends
    when leaving on the left (reward 0) or on the right (reward 1). Returns the list of (state, reward, next state),
    the next state being None at the end. True values (no discount): (s + 1) / (n + 1).
    """
    s, transitions = n // 2, []
    while True:
        s2 = s + (1 if rng.random() < 0.5 else -1)
        if s2 < 0 or s2 >= n:
            transitions.append((s, 1.0 if s2 >= n else 0.0, None))
            return transitions
        transitions.append((s, 0.0, s2))
        s = s2


if __name__ == '__main__':
    P, R = tabular_model(0.1)
    assert np.allclose(P.sum(axis=2), 1)
    env = make_env(slip=0.1, max_steps=200)
    s, _ = env.reset(seed=0)
    total, done = 0, False
    while not done:
        s, r, terminated, truncated, _ = env.step(env.action_space.sample())
        total, done = total + r, terminated or truncated
    print('random policy, one episode of at most 200 steps: return', total)
    print('optimal path', len(OPTIMAL_PATH) - 1, 'steps; safe path', len(SAFE_PATH) - 1, 'steps')
    print('random walk episode:', random_walk_episode(np.random.default_rng(0)))
    print('RL.py: ok')
