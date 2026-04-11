import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error , r2_score

#-----------------------IMPORT---------------------------#
df = pd.read_csv('Boston.csv')

#------------------------Split---------------------------#
X = df.drop('medv', axis=1)
y = df['medv']

#------------------------Train-test-split----------------#
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    #---------------Model-------------------------#
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42

    )
    #-------------------Training------------------#
    model.fit(train_x, train_y)
    #--------------------Predicting---------------#
    y_pred = model.predict(test_x)
    #---------------------Printing----------------#
    print(f"Regression MSE: {mean_squared_error(test_y, y_pred)}")
    print(f"Regression R2: {r2_score(test_y, y_pred):.4f}")

if __name__ == '__main__':
    main()