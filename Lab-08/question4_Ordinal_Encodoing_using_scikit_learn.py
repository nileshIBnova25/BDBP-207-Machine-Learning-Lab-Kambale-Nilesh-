#-----------Ordinal Encoding using scikit-----------------------#

#--------------------IMPORTS------------------------------------#
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder ,OneHotEncoder,LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
from sklearn.impute import SimpleImputer
#---------------------------------------------------------------#

#-------------------Step 1 Loading Data--------------------------------------------#
df= pd.read_csv('breast-cancer_2.csv')

#---------------------Step 2 Separating Features & target------------------------------------------#
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

imputer= SimpleImputer(strategy="most_frequent")
X=imputer.fit_transform(X)
#---------------------------------------------------------------#


def main():
    # -------------------Step3 Encoding X & y --------------------------------#
    ord_enc=OneHotEncoder()
    X_enc = ord_enc.fit_transform(X)
    lab_enc=LabelEncoder()
    y_enc = lab_enc.fit_transform(y)

    #--------------------Step4 Train_test_split------------------------------------------#
    X_train,X_test,y_train,y_test=train_test_split(X_enc,y_enc,test_size=0.3,random_state=42)


    #--------------------Step5 train logistic Regression-------------------------------------------#
    model=LogisticRegression()
    model.fit(X_train,y_train)

    #--------------------Step6 Predicting-------------------------------------------#
    y_pred = model.predict(X_test)


    #-------------------Step7 Evaluation--------------------------------------------#
    print("Accuracy: ",accuracy_score(y_test,y_pred))
    print("\nClassification Report:\n",classification_report(y_test,y_pred))

    #---------------------------------------------------------------#

if __name__ == "__main__" :
    main()