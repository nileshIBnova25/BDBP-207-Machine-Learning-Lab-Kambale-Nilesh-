import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor ,plot_tree
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt

#------------------------Loading Data ------------------------------#
df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
print("Columns in data",df.columns.tolist())
#-------------------------------------------------------------------#

#-----------------------------Separating Features & target--------------------------------------#
X=df.drop(columns=['disease_score', 'disease_score_fluct'])
y=df['disease_score']
#another method
# X=df.iloc[:,0:5]
# y=df.iloc[:,5]
#-----------------------------------------------------------------------------------------------#


def main():
    #----------------------Splitting Into Train-Test----------------------------------------------#
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3,random_state = 42)

    #----------------------Training Model--------------------------------------------------------------#
    model =DecisionTreeRegressor(random_state=42)
    model.fit(X_train,y_train)

    #-------------------------Making Prediction-------------------------------------------------------#
    y_pred = model.predict(X_test)
    #---------------------------Evaluate Model-------------------------#
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

    #---------------------------Visualize Decision Tree----------------#
    plt.figure(figsize=(12,8))
    plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
    plt.show()

if __name__ == "__main__":
    main()

