# ==================Imports=====================#
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==============================================#

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # for Decision Node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, min_samples_leaf=2, max_depth=5):
        self.root = None
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

    def build_tree(self, data, curr_depth=0):
        """Recursively builds a decision tree"""
        X, y = data[:, :-1], data[:, -1]
        num_samples, num_features = X.shape

        # Check stopping conditions: depth and minimum samples
        if num_samples >= self.min_samples_leaf and curr_depth <= self.max_depth:
            # Find the best split
            best_split = self.get_best_split(data, num_samples, num_features)

            # Check if information gain is positive
            if best_split.get("info_gain", 0) > 0:
                # Recur left and right
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)

                # Return decision node
                return Node(
                    best_split["feature_index"],
                    best_split["threshold"],
                    left_subtree,
                    right_subtree,
                    best_split["info_gain"]
                )

        # If conditions for splitting aren't met, return a leaf node
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)

    def get_best_split(self, data, num_samples, num_features):
        """Function to find the best split for the decision tree"""
        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = data[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(data, feature_index, threshold)

                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = data[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")

                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split

    def split(self, data, feature_index, threshold):
        """Optimized split using NumPy indexing"""
        left_indices = data[:, feature_index] <= threshold
        right_indices = data[:, feature_index] > threshold
        return data[left_indices], data[right_indices]

    def information_gain(self, parent, l_child, r_child, mode="gini"):
        """Function to compute the information gain"""
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += (p_cls ** 2)
        return 1 - gini

    def calculate_leaf_value(self, y):
        y = list(y)
        if len(y) == 0: return None
        return max(y, key=y.count)

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(indent + "Class:", tree.value)
        else:
            print(f"feat_{tree.feature_index} <= {tree.threshold} ? Gain: {round(tree.info_gain, 4)}")
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, y):
        data = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(data)

    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value

        feature_val = x[int(tree.feature_index)]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


def main():
    #============================= Data Loading =====================================#
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #======================= Initialize and Train ======================================#
    clf = DecisionTreeClassifier(min_samples_leaf=3, max_depth=3)
    clf.fit(X_train, y_train)

    #============================ Visual Output ========================================#
    print("--- Decision Tree Structure ---")
    clf.print_tree()
    print("-" * 30)

    #============================= Prediction and Evaluation ===========================#
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()