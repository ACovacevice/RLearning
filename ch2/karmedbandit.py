
# coding: utf-8

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt # Required version 3.0.3 or later

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class KArmedBandit:
    
    """
    The k-armed bandit model.
    Rewards come from randomized gaussian distributions.
    """
    
    def __init__(self, k, mean=0.0, variance=1.0, save=False):
        self.k = k
        self.mean_values = np.random.normal(mean, variance, size=k)
        self.save = save
        # Plots reward distributions
        mpl.rcParams["font.size"] = 12
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True, sharex=True, sharey=True)
        plots = ax.violinplot([np.random.normal(loc=self.mean_values[i], size=1000) for i in range(k)])
        for i, body in enumerate(plots["bodies"]):
            body.set_facecolor(colors[i - 10 if k > 10 else i])
            body.set_alpha(0.5)
        ax.set_xticks(range(1, k+1))
        ax.set_xticklabels(["%i" % i for i in range(1, k+1)])
        fig.suptitle("The %i-armed bandit" % k)
        ax.set_xlabel("Action")
        ax.set_ylabel("Reward distribution")
        if save:
            fig.savefig(os.path.join(save, "Figure_2.1.png"), dpi=120)
        else:    
            plt.show()
    
    def reset(self):
        # Resets counters and rewards
        self.avg_reward = 0
        self.Q = np.zeros(shape=(self.k,))
        self.N = np.zeros(shape=(self.k,))
        self.avg_rewards = list()
        self.optimal = np.array([])
    
    def get_action(self, p, step, ucb):
        
        # The epsilon-greedy (exploring) action
        if np.random.rand() < p:
            return np.random.choice(list(range(self.k)))
        
        # The Upper-Confidence-Bound action
        if ucb > 0:
            best_q = self.Q + ucb * np.sqrt(np.log(step) / (self.N + 1e-6))
            return np.random.choice(np.where(best_q == best_q.max())[0])
        
        # The greedy action
        return np.random.choice(np.where(self.Q == self.Q.max())[0])
    
    def get_reward(self, action, step):
        # Computes the reward
        reward = np.random.normal(loc=self.mean_values[action], scale=1.0)
        
        # Updates action counts and action values
        self.N[action] += 1
        step_size = 1 / self.N[action] if self.alpha is None else self.alpha
        self.Q[action] += step_size * (reward - self.Q[action])
        
        # Updates the average reward
        self.avg_reward += (reward - self.avg_reward) / step
        self.avg_rewards.append(self.avg_reward)
        
        # Updates optimal action count
        if action == self.mean_values.argmax():
            self.optimal = np.append(self.optimal, 1)
        else:
            self.optimal = np.append(self.optimal, 0)
    
    def rob(self, p, steps=1000, ucb=0, initial=0):
        # Plays for a certain number of steps
        self.reset()
        self.Q += initial
        t = 1
        while t <= steps:
            action = self.get_action(p, t, ucb)
            self.get_reward(action, t)
            t += 1
            
    def rob_n_times(self, n=100, steps=1000, p=0.1, alpha=None, ucb=0, initial=0):
        # Plays n times a certain number of steps
        self.alpha = alpha
        for robbery in range(n):
            self.rob(p, steps, ucb, initial)
            if robbery == 0:
                avg_R = np.array(self.avg_rewards)
                avg_optimal = self.optimal
            else:
                avg_R += np.array(self.avg_rewards)
                avg_optimal += self.optimal
        return avg_R / n, avg_optimal / n

    def plot_figure_2_2(self):
        # Reproduces and plots the computation of Figure 2.2 in the book. 
        mpl.rcParams["font.size"] = 12
        q0, opt0 = self.rob_n_times(n=2000, steps=1000, p=0.1)
        q1, opt1 = self.rob_n_times(n=2000, steps=1000, p=0.01)
        q2, opt2 = self.rob_n_times(n=2000, steps=1000, p=0.0)
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharex=True)
        ax.plot(q0, color="blue", label="$\epsilon = 0.10$")
        ax.plot(q1, color="red", label="$\epsilon = 0.01$")
        ax.plot(q2, color="green", label="$\epsilon = 0.00$")
        ax1.plot(100 * opt0, color="blue")
        ax1.plot(100 * opt1, color="red")
        ax1.plot(100 * opt2, color="green")
        ax.set_xlabel("Steps"), ax1.set_xlabel("Steps")
        ax.set_ylabel("Average Reward"), ax1.set_ylabel("Optimal Action (%)")
        ax.legend()
        if self.save:
            fig.savefig(os.path.join(self.save, "Figure_2.2.png"), dpi=120)
        else:    
            plt.show()
        
    def plot_figure_2_3(self):
        # Reproduces and plots the computation of Figure 2.3 in the book: optimistic initial values.
        mpl.rcParams["font.size"] = 12
        q0, opt0 = self.rob_n_times(n=2000, steps=1000, p=0.0, alpha=0.1, initial=5)
        q1, opt1 = self.rob_n_times(n=2000, steps=1000, p=0.1, alpha=0.1)
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharex=True)
        ax.plot(q0, color="blue", label="$\\epsilon = 0.0$, Q+5")
        ax.plot(q1, color="red", label="$\\epsilon = 0.1$, Q+0")
        ax1.plot(100 * opt0, color="blue")
        ax1.plot(100 * opt1, color="red")
        ax.set_xlabel("Steps"), ax1.set_xlabel("Steps")
        ax.set_ylabel("Average Reward"), ax1.set_ylabel("Optimal Action (%)")
        ax.legend()
        if self.save:
            fig.savefig(os.path.join(self.save, "Figure_2.3.png"), dpi=120)
        else:    
            plt.show()
    
    def plot_figure_2_4(self):
        # Reproduces and plots the computation of Figure 2.4 in the book: upper-confidence bound.    
        mpl.rcParams["font.size"] = 12
        q0, opt0 = self.rob_n_times(n=2000, steps=1000, p=0.1)
        q1, opt1 = self.rob_n_times(n=2000, steps=1000, p=0.0, ucb=2.0)
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharex=True)
        ax.plot(q0, color="grey", label="$\\epsilon = 0.1$, $UCB = 0.0$")
        ax.plot(q1, color="blue", label="$\\epsilon = 0.0$, $UCB = 2.0$")
        ax1.plot(100 * opt0, color="grey")
        ax1.plot(100 * opt1, color="blue")
        ax.set_xlabel("Steps"), ax1.set_xlabel("Steps")
        ax.set_ylabel("Average Reward"), ax1.set_ylabel("Optimal Action (%)")
        ax.legend()
        if self.save:
            fig.savefig(os.path.join(self.save, "Figure_2.4.png"), dpi=120)
        else:    
            plt.show()
        
if __name__ == "__main__":
    bandit = KArmedBandit(k=10, save=os.path.join("..", "img", "ch2"))
    bandit.plot_figure_2_2()
    bandit.plot_figure_2_3()
    bandit.plot_figure_2_4()
