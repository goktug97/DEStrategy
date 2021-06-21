import unittest
from functools import lru_cache
from enum import IntEnum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm 
from matplotlib.animation import MovieWriter, FFMpegWriter
import seaborn as sns


class Strategy(IntEnum):
    scaledbest1bin = 0
    rand1bin = 1
    best1bin = 2
    rand2bin = 3
    best2bin = 4
    currenttobest1bin = 5
    randtobest1bin = 6
    scaledrand1bin = 7


def arange(lower, upper):
    def _range(f):
        f.range = (lower, upper)
        return f
    return _range


@arange(-5.0, 5.0)
def ackley(x, y):
    first_term = -20 * np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))
    second_term = -np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.e + 20
    return second_term + first_term


@arange(-5.12, 5.12)
def rastrigin(x, y):
    return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)


@arange(-100.0, 100.0)
def schaffer(x, y):
    return 0.5 + ((np.sin(x**2-y**2)**2 - 0.5) / (1 + 0.001 * (x**2+y**2))**2)


@lru_cache(maxsize=1)
def _center_function(population_size):
    centers = np.arange(0, population_size, dtype=np.float32)
    centers = centers / (population_size - 1)
    centers -= 0.5
    centers *= 2.0
    return centers


def _compute_ranks(rewards):
    rewards = np.array(rewards)
    ranks = np.empty(rewards.size, dtype=int)
    ranks[rewards.argsort()] = np.arange(rewards.size)
    return ranks


def rank_transformation(rewards):
    ranks = _compute_ranks(rewards)
    values = _center_function(rewards.size)
    return values[ranks]


def evaluate_population(population, func=ackley):
    return -func(population[:, 0], population[:, 1])


class TestOptimum(unittest.TestCase):
    def test_ackley(self):
        self.assertEqual(ackley(0, 0), 0)

    def test_rastrigin(self):
        self.assertEqual(rastrigin(0, 0), 0)

    def test_schaffer(self):
        self.assertEqual(schaffer(0, 0), 0)


if __name__ == '__main__':
    VISUALIZE = False
    PLOT = True
    MUTATION_FACTOR = 0.7
    CROSSOVER_PROBABILITY = 0.5
    POPULATION_SIZE = 128

    FUNC = ackley
    # FUNC = rastrigin
    # FUNC = schaffer

    if VISUALIZE:
        X = np.linspace(*FUNC.range, 100)     
        Y = np.linspace(*FUNC.range, 100)     
        X, Y = np.meshgrid(X, Y) 
        Z = FUNC(X, Y)
        fig3d, ax3d = plt.subplots(subplot_kw={"projection": "3d"})
        moviewriter = FFMpegWriter()
        moviewriter.setup(fig3d, 'animation.mp4', dpi=100)

    STEPS = 100
    SEEDS = [7235, 4050, 5935, 2919, 2740, 7210, 4012, 5936, 2920, 2741]
    data = []
    for strategy in Strategy:
        for seed in SEEDS:
            np.random.seed(seed)
            population = np.random.uniform(*FUNC.range, (POPULATION_SIZE, 2))
            rewards = evaluate_population(population, FUNC)

            for i in range(STEPS):
                candidate_population = []
                for j in range(POPULATION_SIZE):
                    best_idx = np.argmax(rewards)
                    if strategy == Strategy.best1bin:
                        idxs = np.random.choice(np.delete(np.arange(POPULATION_SIZE), j), 2, replace=False)
                        p0, p1 = population[idxs]
                        diff = p0 - p1
                        p = population[best_idx]
                    elif strategy == Strategy.rand1bin:
                        idxs = np.random.choice(np.delete(np.arange(POPULATION_SIZE), j), 3, replace=False)
                        p0, p1, p2 = population[idxs]
                        diff = p1 - p2
                        p = p0
                    elif strategy == Strategy.rand2bin:
                        idxs = np.random.choice(np.delete(np.arange(POPULATION_SIZE), j), 5, replace=False)
                        p0, p1, p2, p3, p4 = population[idxs]
                        diff = p1 - p2 + p3 - p4
                        p = p0
                    elif strategy == Strategy.best2bin:
                        idxs = np.random.choice(np.delete(np.arange(POPULATION_SIZE), j), 4, replace=False)
                        p1, p2, p3, p4 = population[idxs]
                        diff = p1 - p2 + p3 - p4
                        p = population[best_idx]
                    elif strategy == Strategy.randtobest1bin:
                        idxs = np.random.choice(np.delete(np.arange(POPULATION_SIZE), j), 3, replace=False)
                        p0, p1, p2 = population[idxs]
                        diff = population[best_idx] - p0 + p1 - p2
                        p = p0
                    elif strategy == Strategy.currenttobest1bin:
                        idxs = np.random.choice(np.delete(np.arange(POPULATION_SIZE), j), 2, replace=False)
                        p0, p1 = population[idxs]
                        diff = population[best_idx] - population[j] + p0 - p1
                        p = population[j]
                    elif strategy == Strategy.scaledbest1bin:
                        idxs = np.random.choice(np.delete(np.arange(POPULATION_SIZE), j), 2, replace=False)
                        sub_rewards = rank_transformation(rewards)[idxs]
                        distances = population[idxs] - population[j]
                        diff = sub_rewards @ distances
                        p = population[best_idx]
                    elif strategy == Strategy.scaledrand1bin:
                        idxs = np.random.choice(np.delete(np.arange(POPULATION_SIZE), j), 3, replace=False)
                        sub_rewards = rank_transformation(rewards)[idxs[1:]]
                        distances = population[idxs[1:]] - population[j]
                        diff = sub_rewards @ distances
                        p = population[idxs[0]]
                    else:
                        raise NotImplementedError

                    mutation_vector = p + MUTATION_FACTOR * diff
                    cross = np.random.rand(2) <= CROSSOVER_PROBABILITY
                    new_candidate = population[j].copy()
                    new_candidate[cross] = mutation_vector[cross]
                    candidate_population.append(new_candidate)

                candidate_population = np.array(candidate_population)
                candidate_rewards = evaluate_population(candidate_population, FUNC)
                condition = candidate_rewards > rewards
                population[condition] = candidate_population[condition]
                rewards[condition] = candidate_rewards[condition]
                res = {'generation': i, 'reward': np.max(rewards), 'strategy': strategy.name, 'seed': seed}
                data.append(res)

                if VISUALIZE:
                    ax3d.cla()
                    ax3d.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.nipy_spectral,
                                      linewidth=0.08, antialiased=True)    
                    p = np.array(population)
                    ax3d.plot(p[:, 0], p[:, 1], 'ro') 
                    plt.draw()
                    moviewriter.grab_frame()
                    plt.pause(0.001)
    if VISUALIZE:
        moviewriter.finish()

    if PLOT:
        data = pd.DataFrame(data)
        fig, ax = plt.subplots()
        ax = sns.lineplot(ax=ax, x='generation', y="reward", data=data, ci='sd', hue='strategy',
                          estimator=getattr(np, 'mean'), linewidth=0.8)
        ax.set(xlabel='Generation', ylabel='Reward')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)).set_draggable(True)
        plt.tight_layout(pad=0.5)
        plt.savefig(f'plot_{FUNC.__name__}_{MUTATION_FACTOR}_{CROSSOVER_PROBABILITY}_{POPULATION_SIZE}.png')

    unittest.main()


