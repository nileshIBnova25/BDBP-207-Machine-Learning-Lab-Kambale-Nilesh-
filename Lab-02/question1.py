# 1)[[1,0,2]
#    [0,1,1]
#    [2,1,0] compute covarience by matrix multiplication
#    [1,1,1]
#    [0,2,1]]
#
import sys
x = [[1,0,2],
    [0,1,1],
    [2,1,0],
    [1,1,1],
    [0,2,1]]

u=[] # I column mean
for i in range(len(x[0])):
    value=0
    for j in range(len(x)):
        value +=x[j][i]
    u.append(value/len(x))


xc = [] # mean-center the data
for i in range(len(x)):
    value=[]
    for j in range(len(x[0])):
        value.append(x[i][j]-u[j])
    xc.append(value)

# tranpose matrix

xct=[]
for i in range(len(xc[0])):
    row=[]
    for j in range(len(xc)):
        row.append(xc[j][i])
    xct.append(row)




#matrix multiplication
def matrix_multi(a,b):
    rows_a= len(a)
    rows_b= len(b)
    cols_a= len(a[0])
    cols_b= len(b[0])
    if cols_a != rows_b:
        raise ValueError("Matrix sizes don't match for the multiplication ")
    result = [[0 for i in range(cols_b)] for j in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k]*b[k][j]
    return result
xctxc = matrix_multi(xct,xc)
print(xctxc)
n=len(x)

covarience_matrix=[]
for i in range(len(x[0])):
    row=[]
    for j in range(len(x[0])):
        H=1/(n-1) * xctxc[i][j]
        row.append(H)
    covarience_matrix.append(row)

print(covarience_matrix)

# con firming result using numpy
import numpy as np
x = np.array([[1,0,2],
    [0,1,1],
    [2,1,0],
    [1,1,1],
    [0,2,1]])
cova_max=np.cov(x.T)
print(cova_max)
