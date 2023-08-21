import seaborn as sns
import matplotlib.pyplot as plt


def plot(X, y, Y_pred):
    X = X.ravel() # equivalent of X.reshape(-1)
    y = y.ravel()
    sns.scatterplot(x=X, y=y, color='blue', label='Data')  # The data
    sns.lineplot(x=X, y=Y_pred.ravel(), color='red', label='Linear Model')  # What our model learned
    # Save the plot using Matplotlib's savefig
    plt.savefig('training-pipeline/plot.png')
