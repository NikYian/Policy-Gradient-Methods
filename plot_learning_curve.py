import matplotlib.pyplot as plt 
import numpy as np


def plot_learning_curve(scores, N=100, figure_file='plot'):
    running_avg = np.convolve(scores, np.ones(N)/N, mode='valid') 
    plt.plot(running_avg)
    plt.title('Running Average of previous 100 scores')
    plt.savefig(figure_file)
