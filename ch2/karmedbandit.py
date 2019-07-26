
# coding: utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt # Required version 3.0.1 or later

mpl.rcParams["font.size"] = 12
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class KArmedBandit:
    
    """
    The k-armed bandit model.
    Each action has probability p of being random, and (1 - p) of being greedy. 
    Rewards come from randomized gaussian distributions.
    """
    
    def __init__(self, k, mean=0, variance=1.0):
        self.k = k
        self.reset()
        self.mean_values = np.random.normal(mean, variance, size=k)
        
        # Plots reward distributions
        fig, ax = plt.subplots(1, k, figsize=(10, 6), constrained_layout=True, sharex=True, sharey=True)
        ax[0].set_ylabel("Reward distribution")
        [ax[i].hist(x=np.random.normal(loc=self.mean_values[i], size=5000), bins=50, 
                    orientation='horizontal', color=colors[i]) for i in range(k)]
        [(ax[i].set_xticks([]), ax[i].set_xlabel("Action %i" % (i+1))) for i in range(k)]
        fig.suptitle("The %i-armed bandit" % k)
        plt.show()
    
    def reset(self):
        # Resets counters and rewards
        self.avg_reward = 0
        self.Q = np.zeros(shape=(self.k,))
        self.N = np.zeros(shape=(self.k,))
        self.avg_rewards = list()
        self.optimal = np.array([])
    
    def get_reward(self, action, step):
        # Computes the reward
        reward = np.random.normal(loc=self.mean_values[action], scale=1.0)
        
        # Updates action counts and action values
        self.N[action] += 1
        self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])
        
        # Updates the average reward
        self.avg_reward += (reward - self.avg_reward) / step
        self.avg_rewards.append(self.avg_reward)
    
    def rob(self, p, steps=1000):
        # Plays for a certain number of steps
        self.reset()
        t = 1
        while t <= steps:
            step = np.random.choice(["greedy", "random"], p=[1-p, p], size=1)
            if step == "greedy":
                # Break any possible ties randomly
                action = np.random.choice([a for a, x in enumerate(self.Q) if x == self.Q.max()])
            elif step == "random":
                action = np.random.choice(list(range(self.k)))
            if action == self.mean_values.argmax():
                self.optimal = np.append(self.optimal, 1)
            else:
                self.optimal = np.append(self.optimal, 0)
            self.get_reward(action, t)
            t += 1
            
    def rob_n_times(self, n=100, steps=1000, p=0.1, verbose=False):
        # Plays n times a certain number of steps
        for robbery in range(n):
            self.rob(p, steps)
            if robbery == 0:
                avg_R = np.array(self.avg_rewards)
                avg_optimal = self.optimal
            else:
                avg_R += np.array(self.avg_rewards)
                avg_optimal += self.optimal
        if verbose:
            fig, ax = plt.subplots(1, 1)
            ax.plot(range(steps), avg_R)
            ax.set_xlabel("Steps"), ax.set_ylabel("Average reward")
            plt.show()
        else:
            return avg_R / n, avg_optimal / n
        
if __name__ == "__main__":
    
    bandit = KArmedBandit(k=10)
    
    q0, opt0 = bandit.rob_n_times(n=2000, steps=1000, p=0.1)
    q1, opt1 = bandit.rob_n_times(n=2000, steps=1000, p=0.01)
    q2, opt2 = bandit.rob_n_times(n=2000, steps=1000, p=0.0)
    
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharex=True)

    ax.plot(q0, color="blue", label="$\epsilon = 0.10$")
    ax.plot(q1, color="red", label="$\epsilon = 0.01$")
    ax.plot(q2, color="green", label="$\epsilon = 0.00$")

    ax1.plot(100 * opt0, color="blue", label="$\epsilon = 0.10$")
    ax1.plot(100 * opt1, color="red", label="$\epsilon = 0.01$")
    ax1.plot(100 * opt2, color="green", label="$\epsilon = 0.00$")

    ax.set_xlabel("Steps"), ax1.set_xlabel("Steps")
    ax.set_ylabel("Average Reward"), ax1.set_ylabel("Optimal Action (%)")
    ax.legend()
    plt.show()
    fig.savefig("../img/ch2/karmedbandit_evaluation.png", dpi=100)