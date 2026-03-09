#-----------------------------------------------------------------#
#-------------------------Import------------------------#
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
#-------------------------------------------------------#

#------------------------Loading Data--&--Preprocessing----------------------------#
df=pd.read_csv('breast_cancer.csv')
print(df.info)
print(df.describe)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(y)
print(X)
#-------------------------------------------------------------------#

#-------------------------Train-Test-Split--------------------------#
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#-------------------------------------------------------------------#

#--------------------------StandardScaler-----------------------------------------#
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#-------------------------------------------------------------------#


#---------------------------Logistic_regression_Function-------------#
def log_reggre(X_train,y_train):
    model=LogisticRegression(penalty='l1',solver='liblinear')
    mf=model.fit(X_train,y_train)
    return mf,model
#-------------------------------------------------------------------#
#-------------------------------------------------------------------#
#-------------------------------------------------------------------#
#-------------------------------------------------------------------#
def main():
    mf,model=log_reggre(X_train,y_train)
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    print(f'Accuracy score is ; {accuracy}')
    print(f'LASSO Thetas ; {model.coef_}')

if __name__ == "__main__":
        main()