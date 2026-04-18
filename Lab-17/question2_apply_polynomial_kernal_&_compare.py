# Implement a polynomial kernel K(a,b) = a[0]**2 * b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2
# * b[1]**2 . Apply this kernel function and evaluate the output for the same x1 and x2
# values. Notice that the result is the same in both scenarios demonstrating the power of
# kernel trick.

#------Import--------------#
import pandas as pd
import numpy as np
#--------------------------#

#-----------Data------------------------#
x1 = pd.DataFrame([[3,6]])
x2 = pd.DataFrame([[10,10]])

#---------------------------------------#


#----------------Transfrom_fun-----------------------#
def transform2(X):
    x0=X.iloc[:,0]
    x1=X.iloc[:,1]

    phi0=(x0**2)
    phi1=x0*x1*(2**0.5)
    phi2=(x1**2)

    return np.column_stack((phi0,phi1,phi2))
#----------------------------------------------------#

#---------------Polynomial-Kernel_fun----------------#
def polynomial_kernal(X1,X2):
    x1=X1.values.tolist()[0]
    x2=X2.values.tolist()[0]
    result = x1[0]**2 * x2[0]**2 + 2*x1[0] * x2[0] * x1[1] * x2[1] + x1[1]**2 * x2[1]**2
    return result
#----------------------------------------------------#


def main():
    phi_x1 = phi_x2 = transform2(x1)
    phi_x2 = transform2(x2)
    dot_x = phi_x1 @ phi_x2.T  #computing dot
    plo_k_res = polynomial_kernal(x1,x2)
    print(f"comparison of polynomial kernal with transform function {dot_x} & {plo_k_res}")

if __name__ == '__main__':
    main()







