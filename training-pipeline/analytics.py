import seaborn as sns
import matplotlib.pyplot as plt


def plot(X, y, Y_pred):
    sns.scatterplot(x=X.ravel(), y=y.ravel(), color='blue', label='Data')  # The data
    sns.lineplot(x=X.ravel(), y=Y_pred.ravel(), color='red', label='Linear Model')  # What our model learned
    # Save the plot using Matplotlib's savefig
    plt.savefig('training-pipeline/plot.png')
