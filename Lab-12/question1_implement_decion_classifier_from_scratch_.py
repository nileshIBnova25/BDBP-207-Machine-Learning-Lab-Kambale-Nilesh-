import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load diabetes dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="target")

df = pd.concat([X, y], axis=1)

# --- Your tree functions ---
def get_split(i, df):
    col = sorted(list(set(df.iloc[:, i])))
    split_val = [(col[j] + col[j + 1]) / 2 for j in range(len(col) - 1)]
    return split_val

def mean_sqr_err(y):
    if len(y) == 0:
        return 0
    return  (np.mean((y - np.mean(y)) ** 2))

def build_decision_tree(df, k=5, depth=0, max_depth=5):
    num_row, num_col = df.shape

    if (num_row <= k) or (df.iloc[:, -1].nunique() <= 1) or (depth >= max_depth):
        label = df.iloc[:, -1].mean()
        return {"leaf": True, "value": label}

    lowest_mse = float('inf')
    best_split = None

    for i in range(num_col - 1):
        split_vals = get_split(i, df)
        for val in split_vals:
            left_df = df[df.iloc[:, i] >= val]
            right_df = df[df.iloc[:, i] < val]

            if len(left_df) == 0 or len(right_df) == 0:
                continue

            y_left = left_df.iloc[:, -1]
            y_right = right_df.iloc[:, -1]

            mse = (len(y_left) / num_row) * mean_sqr_err(y_left) + (len(y_right) / num_row) * mean_sqr_err(y_right)

            if mse < lowest_mse:
                lowest_mse = mse
                best_split = {
                    "feature_idx": i,
                    "best_val": val,
                    "left_df": left_df,
                    "right_df": right_df
                }

    if best_split is None:
        return {"leaf": True, "value": df.iloc[:, -1].mean()}

    return {
        "leaf": False,
        "feature_idx": best_split["feature_idx"],
        "best_val": best_split["best_val"],
        "left": build_decision_tree(best_split["left_df"], k, depth + 1, max_depth),
        "right": build_decision_tree(best_split["right_df"], k, depth + 1, max_depth)
    }

def predict_tree(tree, x):
    if tree["leaf"]:
        return tree["value"]
    feature_val = x[tree["feature_idx"]]
    if feature_val >= tree["best_val"]:
        return predict_tree(tree["left"], x)
    else:
        return predict_tree(tree["right"], x)

def predict(tree, X):
    return np.array([predict_tree(tree, x) for x in X.to_numpy()])

# --- Train/Test Split ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


def main():
    # Build tree
    tree = build_decision_tree(train_df, k=60, max_depth=10)

    # Predict
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    y_pred = predict(tree, X_test)

    # R^2 score
    r2 = r2_score(y_test, y_pred)
    print("R^2 score:", r2)
    print("Decision Tree Structure:\n", tree)

if __name__ == "__main__":
    main()










# import pandas as pd
# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor, plot_tree
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
#
# #------------------------Loading Data ------------------------------#
# diabetes = load_diabetes()
# X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# y = pd.Series(diabetes.target, name="disease_score")  # rename target similar to your example
# print("Columns in data:", X.columns.tolist())
# #-------------------------------------------------------------------#
#
# def main():
#     #----------------------Splitting Into Train-Test----------------------------------------------#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
#     print(y_train)
#     print(y_test)
#
#     #----------------------Training Model--------------------------------------------------------------#
#     model = DecisionTreeRegressor(
#         random_state=999,
#         max_depth=10,  # limit depth
#         min_samples_leaf=60 # minimum samples per leaf
#     )
#     model.fit(X_train, y_train)
#
#     #-------------------------Making Prediction-------------------------------------------------------#
#     y_pred = model.predict(X_test)
#
#     #---------------------------Evaluate Model-------------------------#
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#
#     print("Mean Squared Error:", mse)
#     print("R^2 Score:", r2)
#
#     #---------------------------Visualize Decision Tree----------------#
#     plt.figure(figsize=(20,12))
#     plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
#     plt.show()
#
# if __name__ == "__main__":
#     main()