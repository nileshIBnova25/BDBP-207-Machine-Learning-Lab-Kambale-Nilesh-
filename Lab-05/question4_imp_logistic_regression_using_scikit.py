#4. Implement logistic regression using scikit-learn for the breast cancer dataset
#--------------------------------Step-1-:-Import-------------------------------------#
# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
#------------------------------------------------------------------------------------#




#----------------------------Step-2-:--Importing-Data--------------------------------#
df=pd.read_csv('data.csv')
df.head()

#------------------Step-3--concise summary of a DataFrame's metadata-----------------#
df.info()

#--Step-4--how much" (distribution) by generating a statistical summary of your data-#
df.describe()

#---------------------------Step-5-Droping-Unnamed-Column-NAN------------------------#
df.drop(columns=['Unnamed: 32','id'],axis=1,inplace=True)

#------------------------------------------------------------------------------------#




#---------------------------Step-6-PLOT-FOR_BENIGN_&_MALIGNANT---------------------------------------#
g=sns.countplot(data=df,x='diagnosis',hue='diagnosis')
for i in g.containers:
    g.bar_label(i)
plt.show()
#----------------------------------------------------------------------------------------------------#

#--------------------------Step-7-PLOTING-HISTOGRAM--FOR-ALL-FEATURES----------------------- --------#
df.hist(figsize=(30,30))
plt.show()
#----------------------------------------------------------------------------------------------------#

#--------------------------------Step-8-Encoding-M-as-1-&-B-as-0-------------------------------------#
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
#----------------------------------------------------------------------------------------------------#

#----------------------Step-9-DROPING-DIAGNOSIS-COLUMN--&-SAVING-IT-IN-TO-X-&-y----------------------#
X=df.drop('diagnosis',axis=1)
y=df['diagnosis']
#----------------------------------------------------------------------------------------------------#


#--------------------------------Step-10-SPLITING-INTO-TRAIN-TEST------------------------------------#
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#----------------------------------------------------------------------------------------------------#



#------------------------------Step-11-to-replace-missing-values-(NaN)-------------------------------#
imputer=SimpleImputer(strategy='mean')


#------------------------Step-12-Calculate-mean-replace-with-missing-value---------------------------#
X_train_imputer=imputer.fit_transform(X_train)
X_test_imputer=imputer.transform(X_test)
#----------------------------------------------------------------------------------------------------#



#-------------------------Step-13-Standardizing-&-scaling--------------------------------------------#
scalar=StandardScaler()
#----------------------------------------------------------------------------------------------------#
X_train_scalar=scalar.fit_transform(X_train_imputer)
#----------------------------------------------------------------------------------------------------#
X_test_scalar=scalar.transform(X_test_imputer)
#----------------------------------------------------------------------------------------------------#








#--------------------------Step-14-Importing-&-Scaling-The-Model-------------------------------------#
logistic_model=LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scalar,y_train)
#----------------------------------------------------------------------------------------------------#

accuracies=cross_val_score(estimator=logistic_model,X=X_train_scalar,y=y_train,cv=5,scoring='accuracy')
print('Accuracy: {:.2f} %'.format(accuracies.mean()*100))
print('Standard Deviation: {:.2f} %'.format(accuracies.std()*100))
#----------------------------------------------------------------------------------------------------#



y_pred=logistic_model.predict(X_test_scalar) # prediction of y
#----------------------------------------------------------------------------------------------------#
acuuracy = accuracy_score(y_test,y_pred)
print('Accuracy: {:.2f} %'.format(acuuracy*100))
#----------------------------------------------------------------------------------------------------#
confusion_matrix(y_test,y_pred)
#----------------------------------------------------------------------------------------------------#
print(classification_report(y_test,y_pred))
#----------------------------------------------------------------------------------------------------#