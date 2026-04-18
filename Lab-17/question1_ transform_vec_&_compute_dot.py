# Let x1 = [3, 6], x2 = [10, 10]. Use the above “Transform” function to transform these
# vectors to a higher dimension and compute the dot product in a higher dimension. Print
# the value.

#------Import--------------#
import pandas as pd
import numpy as np
#--------------------------#

#-----------Data------------------------#
x1 = pd.DataFrame([[3,6]])
x2 = pd.DataFrame([[10,10]])

#---------------------------------------#
print(x1,x2)


#----------------Transfrom_fun-----------------------#
def transform2(X):
    x0=X.iloc[:,0]
    x1=X.iloc[:,1]

    phi0=(x0**2)
    phi1=x0*x1*(2**0.5)
    phi2=(x1**2)

    return np.column_stack((phi0,phi1,phi2))
#----------------------------------------------------#

def main():
    phi_x1 = phi_x2 = transform2(x1)
    phi_x2 = transform2(x2)
    dot_x = phi_x1 @ phi_x2.T  #computing dot
    print(dot_x)
if __name__ == '__main__':
    main()



