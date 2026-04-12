import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


#------------------------Loading-Data-------------------------------#
df = load_diabetes()
X = df.data
y = df.target

#----------------------Split-Train-Test--------------------------#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

def create_multiple_trees(n_trees,X_train,y_train):

    trees = [] # for storing multiple trees

    for i in range(n_trees):
        # creating a boostrap sample (random sampling with replacement)
        indices = np.random.choice(len(X_train),size=len(X_train),replace = True)
        X_boostrap , y_boostrap = X_train[indices], y_train[indices]

        #intialize & fit single tree
        tree = DecisionTreeRegressor(max_depth=5,random_state=i)
        tree.fit(X_boostrap, y_boostrap)
        trees.append(tree)

    return trees

def get_aggrigated_trees(models,data_input):
    #collect prediction from every tree
    all_pred = np.array([model.predict(data_input) for model in models])

    final_pred = np.mean(all_pred, axis=0)
    return final_pred



def main():
    trees = create_multiple_trees(n_trees=10,X_train=X_train,y_train=y_train)
    y_pred = get_aggrigated_trees(trees,X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error of Aggregated Trees : {mse:.2f}")

    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score of Aggregated Tree : {r2:.2f}")

    single_tree_mse = mean_squared_error(y_test, trees[0].predict(X_test))
    print(f"Mean Squared Error of Single Trees : {single_tree_mse:.2f}")

    single_tree_r2 = r2_score(y_test, trees[0].predict(X_test))
    print(f"R2 Score of Single Trees : {single_tree_r2:.2f}")


if __name__ == '__main__':
    main()








