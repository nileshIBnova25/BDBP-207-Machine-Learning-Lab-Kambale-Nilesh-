#------------------IMPORT-----------------------------#
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as m
#-----------------------------------------------------#

#------------------Step1-Loading-Data-----------------#
df = pd.read_csv('transform.csv')
df['Label'] = df['Label'].map({'Blue':1,'Red':0})
X=df.iloc[:, :-1]
y=df['Label']
#-----------------------------------------------------#

#----------------Transform-Function-------------------#

def transform2(X):
    x0=X.iloc[:,0]
    x1=X.iloc[:,1]

    phi0=(x0**2)
    phi1=x0*x1*(2**0.5)
    phi2=(x1**2)

    return np.column_stack((phi0,phi1,phi2))

#------------------------------------------------------#


#--------------------2D-Plot---------------------------#
def scatter_plot_2d(X, y):
    x0 = X.iloc[:, 0].values
    x1 = X.iloc[:, 1].values

    plt.figure(figsize=(6, 5))

    for label in np.unique(y):
        mask = (y == label)
        plt.scatter(x0[mask], x1[mask], label=f'Class {label}', alpha=0.7)

    plt.xlabel('Feature x0')
    plt.ylabel('Feature x1')
    plt.title('2D Feature Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
#------------------------------------------------------#

#-------------------3D-Plot----------------------------#
def scatter_plot_3d(data, y):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # One-liner scatter: 'c' handles colors, 'cmap' chooses the palette
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=y, cmap='coolwarm', edgecolors='k')

    # Simpler Plane: Use only 2 points for the grid to keep it lightweight
    x_range = np.array([data[:, 0].min(), data[:, 0].max()])
    y_range = np.array([data[:, 1].min(), data[:, 1].max()])
    xx, yy = np.meshgrid(x_range, y_range)
    zz = 300 - 0.5 * xx - 4.5 * yy

    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

    ax.set_xlabel("x1²");
    ax.set_ylabel("√2 x1x2");
    ax.set_zlabel("x2²")
    plt.show()
#--------------------------------------------------------#


def main():
    phi_x = transform2(X)
    print(phi_x)
    scatter_plot_2d(X,y)
    scatter_plot_3d(phi_x,y)

if __name__ == '__main__':
    main()







