import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Dataset Setup
data = {
    'x1': [6, 6, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14],
    'x2': [5, 9, 6, 8, 10, 2, 5, 10, 13, 5, 8, 6, 11, 4, 8],
    'Label': [0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0]
}
df = pd.DataFrame(data)
X, y = df[['x1', 'x2']], df['Label']


def plot_decision_boundaries(X, y, kernel, **kwargs):
    # Fit inside the plotter to ensure the meshgrid matches the model's expected input
    model = SVC(kernel=kernel, **kwargs).fit(X, y)

    # Create grid
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Predict on meshgrid
    # We use pd.DataFrame here to ensure the model sees feature names 'x1' and 'x2'
    grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)
    Z = model.predict(grid_df).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title(f"SVM with {kernel.upper()} Kernel")
    plt.show()


def main():
    # --- Polynomial Kernel (Degree 2) ---
    poly_svc = SVC(kernel='poly', degree=2, coef0=1, C=1.0).fit(X, y)
    y_pred_poly = poly_svc.predict(X)

    # --- RBF Kernel ---
    rbf_svc = SVC(kernel='rbf', gamma=0.5).fit(X, y)
    y_pred_rbf = rbf_svc.predict(X)

    # --- Results ---
    print(f"Polynomial Accuracy: {accuracy_score(y, y_pred_poly) * 100:.1f}%")
    print(f"RBF Accuracy:        {accuracy_score(y, y_pred_rbf) * 100:.1f}%")

    # Visualizations
    plot_decision_boundaries(X, y, kernel='poly', degree=2, coef0=1)
    plot_decision_boundaries(X, y, kernel='rbf', gamma=0.5)


if __name__ == '__main__':
    main()