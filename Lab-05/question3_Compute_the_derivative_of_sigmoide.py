#3 Compute the derivative of a sigmoid function and visualize it
#-------------Import-----------------------------------#
import math

import matplotlib.pyplot as plt
#------------------------------------------------------#

#-------------The-Sigmoid-Function---------------------#

x=[i for i in range(-30,30,1)]     # Creation of vector
def g_of_z(x):                     # Function
    return 1/(1+math.exp(-x))

#-------------Derivative-Function------------------ ---#
def derv_fun(x):
    lis=[]
    for i in range(len(x)):
        lis.append(g_of_z(x[i])*(1-(g_of_z(x[i]))))
    return lis
#------------------------------------------------------#


#------------------------------------------------------#
def main():
    y=derv_fun(x)
    #------Derivative-Ploting-Sigmide_function---------#
    plt.plot(x,y)
    plt.xlabel('Vector x')
    plt.ylabel('Vector Derivative Function Value')
    plt.title('Derivative-Of-Sigmoid Function')
    plt.grid(True)
    plt.show()
    #___________________________________________________#


    print(derv_fun(x))
    print(x)

if __name__ == '__main__':
    main()
