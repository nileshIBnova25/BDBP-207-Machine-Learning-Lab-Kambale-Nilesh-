#-------------------------Imports-----------------------------------#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#-------------------------------------------------------------------#
#----------------------------------Loading-Of-Data---------------------------------------#

df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
df.plot()
X=df.iloc[:,0:5]

y=df.iloc[:,6]

#----------------------------Dividing-into-train-test-----------------------------------#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

n,d=X_train.shape

#---------------------------------------------------------------------------------------#

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#-----------------------Adding-Intetcept-&-Creating-Theta-------------------------------#

def add_intercept(X_train):
    x0=np.ones((X_train.shape[0],1))
    X_train_add=np.hstack((x0,X_train))
    return X_train_add


def cr_theta(X_train_inc):
    theta=[0]*X_train_inc.shape[1]
    return theta

X_train_inc=(add_intercept(X_train))
theta=cr_theta(X_train_inc)
X_test_inc=(add_intercept(X_test))

#--------------------------------Converting-Data-Frame-Into-List-------------------------------#
x=X_train_inc.tolist()
y=y_train.tolist()
x_test=X_test_inc.tolist()
y_test=y_test.tolist()

#------------------------------------Function-Being-Used---------------------------------------#
def make_inverse(X):  # step 1 : creation of transpose matrix
    xt = []
    for i in range(len(X[0])):
        row = []
        for j in range(len(X)):
            row.append(X[j][i])
        xt.append(row)
    return xt


def matrix_multplication(a, b):
    rows_a=len(a)
    rows_b=len(b)
    cols_a=len(a[0])
    cols_b=len(b[0])
    if cols_a!=rows_b:
        raise ValueError('Number of cols of frist matrix must be similer to No. rows of second matrix' )
    result=[[0 for i in range(cols_b)] for j in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j]+=a[i][k]*b[k][j]
    return result


def inversext(x):
    inv = np.linalg.inv(x)
    return inv

#----------------------------Matrix-Multiplication-For-Predicting-Y--------------------------------#
def mat_vect_multiply(a, b):
       rows_a = len(a)
       cols_a = len(a[0])

       if cols_a != len(b):
           raise ValueError("Number of columns of a must match size of vector b")

       result = [0] * rows_a

       for i in range(rows_a):
           for k in range(cols_a):
               result[i] += a[i][k] * b[k]

       return result

#-------------------------------------------------------------------------------------------------#
#-------------------------------R2 SCORE CALCULATION FUNCTION-------------------------------------#

def r2score(y_test, y_pred):
    mean_y= sum(y_test) / len(y_test)
    sum_of_square_test = sum((y_test[i] - mean_y) ** 2 for i in range(len(y_test)))
    sum_of_square_res = sum((y_test[i] - y_pred[i]) ** 2 for i in range(len(y_test)))
    return 1 - (sum_of_square_res/sum_of_square_test)

#-------------------------------------------------------------------------------------------------#


#------------------------------------Main-Function------------------------------------------------#

def main():
    def normal_equation(x,y):
        xt=make_inverse(x)                                  # step 1 : creation of transpose matrix
        xtx=matrix_multplication(xt,x)                      # step 2 : Multiplying Xt wih X
        xtxa=np.array(xtx)                                  #-----List-To-Array-----#
        xtx_inva=inversext(xtxa)                            # step 3 : Making XTX to (XTX)inverse
        xtx_inv=xtx_inva.tolist()                           #-----Array-To-List-----#
        xtx_inv_xt=matrix_multplication(xtx_inv,xt)         # step 4 : Multiplying again with XT
        theta_result=mat_vect_multiply(xtx_inv_xt,y)        # step 5 : Multiplying with y
        return theta_result                                 #----------------theta----------------#

    theta_res=normal_equation(x,y)
    y_predict_res=mat_vect_multiply(x_test,theta_res)
    r2score_res=r2score(y_test,y_predict_res)

    print(r2score_res)

if __name__ == "__main__":
    main()
