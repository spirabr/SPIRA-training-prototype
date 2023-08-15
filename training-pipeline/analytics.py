import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting


def plot(a):
    plt.hist(a)
    plt.savefig('training-pipeline/plot.png')  # Save the plot as an image file
