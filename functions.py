import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_map(y_true, y_pred, labels, title):
    """Show the confusion matrix as a heatmap (rows = true, columns = predicted)."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(matrix, annot=True, cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()
