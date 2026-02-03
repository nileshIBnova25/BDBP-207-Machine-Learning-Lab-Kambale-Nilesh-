#-------------Import-----------------------------------#
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#------------------------------------------------------#

#======================================================#
#-------------------- set parameters here--------------#
col = 5             #                                  #      #
alpha = 1e-3        #                                  #      # learning rate
alpha2 = 1e-5
iteration = 1000    #                                  #      # No of iteraton for large data if nned to increase
tol = 0.001         #                                  #      # Toleration limit fopr diffrence between 2 theta
#------------------------------------------------------#
#======================================================#



#---------------------------------------------------------------------------------------#
# 1)Load Data Set
# a) Simulated data
df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
df.plot()
X=df.iloc[:,0:5]
y1=df.iloc[:,col]   # disease fluct
y2=df.iloc[:,col+1]   # disease fluct


# b) California Housing
def load_data():
    [X,y] = fetch_california_housing(return_X_y=True) # 1)Load Data Set
    return X,y
X1,Y1=load_data()


#---------------------------------------------------------------------------------------#
#2) Divide it into train-test
# a) Simulated data
# i)  Disese score
X_train_sc0, X_test_sc0, y_train_sc0, y_test_sc0 = train_test_split(X, y1, test_size=0.3, random_state=999)
# ii) Disese score fluct
X_train_fl0, X_test_fl0, y_train_fl0, y_test_fl0 = train_test_split(X, y2, test_size=0.3, random_state=999)
n,d=X_train_sc0.shape

# b)California Housing
X_train_cal0, X_test_cal0, y_train_cal0, y_test_cal0 = train_test_split(X1, Y1, test_size=0.3, random_state=999)

#---------------------------------------------------------------------------------------#

scaler=StandardScaler()
X_train_sc=scaler.fit_transform(X_train_sc0)
X_test_sc=scaler.transform(X_test_sc0)
X_train_fl=scaler.fit_transform(X_train_fl0)
X_test_fl=scaler.transform(X_test_fl0)
X_train_cal=scaler.fit_transform(X_train_cal0)
X_test_cal=scaler.fit_transform(X_test_cal0)


#---------------------------------------------------------------------------------------#
def add_intercept(X_train):
    x0=np.ones((X_train.shape[0],1))
    X_train_add=np.hstack((x0,X_train))
    return X_train_add


def cr_theta(X_train_inc):
    theta=[0]*X_train_inc.shape[1]
    return theta

X_train_sc = (add_intercept(X_train_sc))
X_test_sc = (add_intercept(X_test_sc))
theta_sc = cr_theta(X_train_sc)

X_test_fl = add_intercept(X_test_fl)
X_train_fl=add_intercept(X_train_fl)
theta_fl=cr_theta(X_train_fl)

X_test_cal=add_intercept(X_test_cal)
X_train_cal=add_intercept(X_train_cal)
theta_cal=cr_theta(X_train_cal)


# changing data frame to list
x1=X_train_sc.tolist()
x_test1=(X_test_sc.tolist())

y1=y_train_sc0.tolist()
y_test1=y_test_sc0.tolist()


x2=X_train_fl.tolist()
x_test2=(X_test_fl.tolist())

y2=y_train_fl0.tolist()
y_test2=y_test_fl0.tolist()


x3=X_train_cal.tolist()
x_test3=(X_test_cal.tolist())

y3=y_train_cal0.tolist()
y_test3=y_test_cal0.tolist()




#----------------------------------------------------------------------------------------#



#----------------------------------------------------------------------------------------#
def hypothesis_fun(x, theta, i2):
    total = 0
    for i1 in range(len(theta)):
        total += x[i2][i1] * theta[i1]
    return total


def deriv_fun(x, y, theta, i3):
    z = 0
    for i2 in range(len(x)):
        z += (hypothesis_fun(x, theta, i2) - y[i2]) * x[i2][i3]
    return z


def cost_fun(x,y,theta):
    J_of_theta = 0
    for i in range(len(x)):
        J_of_theta += (hypothesis_fun(x, theta, i) - y[i]) ** 2
    return J_of_theta
#---------------------------------------------------------------------------------------#


#-------------------------R2 SCORE CALCULATION FUNCTION---------------------------------#

def r2score(y_test, y_pred):
    mean_y= sum(y_test) / len(y_test)
    sum_of_square_test = sum((y_test[i] - mean_y) ** 2 for i in range(len(y_test)))
    sum_of_square_res = sum((y_test[i] - y_pred[i]) ** 2 for i in range(len(y_test)))
    return 1 - (sum_of_square_res/sum_of_square_test)

#---------------------------------------------------------------------------------------#


#---------------------------------------------------------------main function---------------------------------#
def main():
    #------------------------Using scikit learn---------------------------------------------------------------#

    def linear_reg_sckit(X_train, y_train, X_test, y_test):
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        scaler = StandardScaler()                     #3)Standardize the data
        scaler = scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scale = scaler.transform(X_test)

        model = LinearRegression()                    #4 Intializing the model

        model.fit(X_train_scaled, y_train)            #5 Training the model

        y_pred = model.predict(X_test_scale)          #6 Test the
        r2 = r2_score(y_test, y_pred)
        return r2

    #-----------------------------Using function defined by me scratch ---------------------------------------------#

    def get_theta(theta, x, y, alpha, iteration, tol):

        for itr in range(iteration):
            new_theta = [0] * len(theta)

            for i3 in range(len(theta)):
                new_theta[i3] = round(theta[i3] - alpha * deriv_fun(x, y, theta, i3),10)

            diff = max(abs(new_theta[i] - theta[i]) for i in range(len(theta)))

            theta = new_theta  # update theta

            j_theta = cost_fun(x, y, theta)

            if diff < tol:
                print(f"Stopped early at iteration {itr}")
                break
        return theta

    def y_predict(a, b):
        rows_a = len(a)
        cols_a = len(a[0])

        if cols_a != len(b):
            raise ValueError("Number of columns of a must match size of vector b")

        result = [0] * rows_a

        for i in range(rows_a):
            for k in range(cols_a):
                result[i] += a[i][k] * b[k]

        return result


    #----------------------------------------Comparision----------------------------------------------------------------------#
    # 1) comparing simulated data
    #  a) Disease score
    r2_score_scikit_disease_score = linear_reg_sckit(X_train_sc0, y_train_sc0, X_test_sc0, y_test_sc0)
    theta_for_disease_score = get_theta(theta_sc, x1, y1, alpha, iteration, tol)
    y_predict_disease_score = y_predict(x_test1,theta_for_disease_score)
    r2_disease_score = r2score(y_test1, y_predict_disease_score)
    print(f"=================================================================================================================")
    print(f"r2 score of disease score by scikit learn : {r2_score_scikit_disease_score}")
    print(f"r2 score of disease score by scratch implementation : {r2_disease_score}")
    print(f"Difference between bot r2 score :{r2_score_scikit_disease_score - r2_disease_score }")

    # b) Disease score fluctuation
    r2_score_scikit_disease_score_fluct = linear_reg_sckit(X_train_fl0, y_train_fl0, X_test_fl0, y_test_fl0)
    theta_for_disease_score_fluct = get_theta(theta_fl, x2, y2, alpha, iteration, tol)
    y_predict_disease_score_fluct = y_predict(x_test2,theta_for_disease_score_fluct)
    r2_disease_score_fluct = r2score(y_test2, y_predict_disease_score_fluct)
    print(f"=================================================================================================================")
    print(f"r2 score of disease score fluct by scikit learn : {r2_score_scikit_disease_score_fluct}")
    print(f"r2 score of disease score fluct by scratch implementation : {r2_disease_score_fluct}")
    print(f"Difference between bot r2 score :{r2_score_scikit_disease_score_fluct - r2_disease_score_fluct }")

    # 2) California Housing Prediction
    r2_score_scikit_california = linear_reg_sckit(X_train_cal0, y_train_cal0, X_test_cal0, y_test_cal0)
    theta_for_cal = get_theta(theta_cal, x3, y3, alpha2, iteration, tol)
    y_predict_cal = y_predict(x_test3,theta_for_cal)
    r2_california = r2score(y_test3, y_predict_cal)
    print(f"=================================================================================================================")
    print(f"r2 score of california by scikit learn : {r2_score_scikit_california}")
    print(f"r2 score of california by scratch implementation : {r2_california}")
    print(f"Difference between bot r2 score :{r2_score_scikit_california - r2_california }")

if __name__ == "__main__":
    main()
#--------------------------------------------------------------------------------------------------------------#