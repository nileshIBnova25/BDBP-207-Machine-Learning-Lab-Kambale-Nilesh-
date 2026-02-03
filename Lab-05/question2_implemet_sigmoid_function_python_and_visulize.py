#2.​ Implement sigmoid function in python and visualize it
#-------------Import-----------------------------------#
import math

import matplotlib.pyplot as plt
#------------------------------------------------------#

#-------------The-Sigmoid-Function---------------------#

x=[i for i in range(-10,10,1)]     # Creation of vector
def g_of_z(x):                     # Function
    return 1/(1+math.exp(-x))
#------------------------------------------------------#



#------------------------------------------------------#
def main():
    def sigmoide_fun(x):
        lis=[]
        for i in range(len(x)):
            d=g_of_z(x[i])
            lis.append(d)
        return lis
    y=sigmoide_fun(x)

    #-----------Ploting-Sigmide_function---------------#
    plt.plot(x,y)
    plt.xlabel('Vector x')
    plt.ylabel('Vector After passing through sigmoide function')
    plt.title('Sigmoid Function')
    plt.grid(True)
    plt.show()
    #___________________________________________________#


    print(sigmoide_fun(x))
    print(x)

if __name__ == '__main__':
    main()
