import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]  # Fixed: removed trailing comma
        x_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[x_column < self.threshold] = -1
        else:
            predictions[x_column > self.threshold] = -1
        return predictions


class Adaboost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            for feature_i in range(n_features):
                x_column = X[:, feature_i]
                thresholds = np.unique(x_column)
                for threshold in thresholds:
                    for p in [1, -1]:  # Check both polarities for every threshold
                        predictions = np.ones(n_samples)
                        if p == 1:
                            predictions[x_column < threshold] = -1
                        else:
                            predictions[x_column > threshold] = -1

                        error = sum(w[y != predictions])

                        if error < min_error:
                            min_error = error
                            clf.polarity = p
                            clf.threshold = threshold
                            clf.feature_idx = feature_i

            # Important: Use min_error (the error of the BEST stump) not 'error'
            eps = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error) / (min_error + eps))

            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)  # Re-normalize weights

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        return np.sign(y_pred)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


# Data Setup
data = load_iris()
X, y = data.data, data.target
y = np.where(y == 0, -1, 1)  # Convert to binary classification

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = Adaboost(n_clf=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")













