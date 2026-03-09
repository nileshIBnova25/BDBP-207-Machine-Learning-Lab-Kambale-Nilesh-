from sklearn.datasets import load_diabetes
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor , plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error
import matplotlib.pyplot as plt

#loading data
dia = load_diabetes()
X = dia.data
y = dia.target

# train test split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2 , random_state=42 )


def main():
    base_estimator = DecisionTreeRegressor(max_depth=5,min_samples_leaf=5)
    bag_reg=BaggingRegressor(estimator=base_estimator,n_estimators=50,random_state=42)
    bag_reg.fit(X_train,y_train)
    y_pred = bag_reg.predict(X_test)
    mse=mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f"Mean Squared Error: ", mse)
    print(f"R2 Score : ", r2)

if __name__ == "__main__":
    main()
