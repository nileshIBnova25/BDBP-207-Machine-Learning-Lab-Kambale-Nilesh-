# 1)Implement a linear regression model using scikit-learn for the simulated dataset -
# simulated_data_multiple_linear_regression_for_ML.csv - to predict the disease score
# from multiple clinical parameters
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# 1)Load Data Set
df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
df.plot()
print(df)
print(df.size)
# df.drop("disease_score_fluct" ,axis=1,inplace=True)
# print(df)
X=df.iloc[:,0:5]
y=df.iloc[:,5]
print(X)
print(y)
#2) Divide it into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

#print(X_train.shape)
#print(X_test.shape)

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





