import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_train = [[1,1,2],[1,2,1],[1,3,3]]
y_train = [3,4,5]
X_test = [[1,2,1],[1,3,3]]
y_test = [4,5]




def main():
    #3)Standardize the data

    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(X_train_scaled.shape)
    print(X_test_scaled.shape)

    #4 Initialization tthe model
    model = LinearRegression()

    #5 Training the model
    model.fit(X_train_scaled, y_train)

    #6 Test the model
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    print(r2)


    print('Done !')
if __name__ == '__main__':
    main()