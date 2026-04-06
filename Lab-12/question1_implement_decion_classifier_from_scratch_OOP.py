import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Node class ---
class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf

# --- Decision Tree Regressor ---
class DecisionTreeRegressor:
    def __init__(self, min_samples_leaf=1, max_depth=6):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.root = None

    def _calculate_mse(self, y):
        if len(y) == 0:
            return 0
        return np.var(y)

    def _get_best_split(self, df):
        num_rows, num_cols = df.shape
        y = df.iloc[:, -1]

        if num_rows <= self.min_samples_leaf:
            return None

        best_mse = float('inf')
        best_feature = None
        best_thresh = None

        for i in range(num_cols - 1):  # exclude target
            col_vals = df.iloc[:, i].unique()
            col_vals.sort()
            thresholds = (col_vals[:-1] + col_vals[1:]) / 2

            for t in thresholds:
                left = df[df.iloc[:, i] <= t]
                right = df[df.iloc[:, i] > t]

                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue

                mse = (len(left)/num_rows)*self._calculate_mse(left.iloc[:, -1]) + \
                      (len(right)/num_rows)*self._calculate_mse(right.iloc[:, -1])

                if mse < best_mse:
                    best_mse = mse
                    best_feature = i
                    best_thresh = t

        if best_feature is None:
            return None
        return best_feature, best_thresh

    def _build_tree(self, df, depth=0):
        y = df.iloc[:, -1]

        # Leaf condition
        if depth >= self.max_depth or len(df) <= self.min_samples_leaf or y.nunique() <= 1:
            return Node(value=y.mean(), is_leaf=True)

        split = self._get_best_split(df)
        if split is None:
            return Node(value=y.mean(), is_leaf=True)

        feature_idx, threshold = split
        left_df = df[df.iloc[:, feature_idx] <= threshold]
        right_df = df[df.iloc[:, feature_idx] > threshold]

        left_child = self._build_tree(left_df, depth+1)
        right_child = self._build_tree(right_df, depth+1)

        return Node(feature_idx=feature_idx, threshold=threshold,
                    left=left_child, right=right_child, is_leaf=False)

    def fit(self, df):
        self.root = self._build_tree(df)

    def _predict_single(self, row, node):
        if node.is_leaf:
            return node.value
        if row.iloc[node.feature_idx] <= node.threshold:
            return self._predict_single(row, node.left)
        else:
            return self._predict_single(row, node.right)

    def predict(self, df):
        return np.array([self._predict_single(row, self.root) for _, row in df.iterrows()])

# --- R² Score ---
def calculate_r2(y_true, y_pred):
    ssr = np.sum((y_true - y_pred)**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ssr/sst

def main():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=df.columns[:-1])
    X_test = pd.DataFrame(scaler.transform(X_test), columns=df.columns[:-1])

    train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

    model = DecisionTreeRegressor(min_samples_leaf=60, max_depth=40)
    model.fit(train_df)

    y_pred_train = model.predict(train_df.iloc[:, :-1])
    y_pred_test = model.predict(test_df.iloc[:, :-1])

    print("--- Sample Predictions on Test Set ---")
    for i in range(5):
        print(f"Sample {i+1}: Predicted={y_pred_test[i]:.2f}, Actual={y_test.values[i]:.2f}")

    r2_test = calculate_r2(y_test.values, y_pred_test)
    print(f"R² Score (Test):  {r2_test:.4f}")

if __name__ == "__main__":
    main()