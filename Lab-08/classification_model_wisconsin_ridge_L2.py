#---------------------classification model wisconsin ridge L2--------------------------------------------------#
#--------------------------IMPORT---------------------------------------------#
import pandas as pd
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
#-----------------------------------------------------------------------#

#-------------------Loading Data--&--Preprocessing------------------#
df = pd.read_csv('breast_cancer.csv')
print(df.info)
print(df.describe)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(y)
print(X)
#-------------------------------------------------------------------#

#-------------------------Train-Test-Split--------------------------#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#-------------------------------------------------------------------#

#--------------------------StandardScaler--------------------------#
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#------------------------------------------------------------------#


def main():
    clf=RidgeClassifier()
    ridge_model=clf.fit(X_train,y_train)
    y_pred=ridge_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    print(f' accuracy : {acc}')

if __name__ == "__main__":
    main()