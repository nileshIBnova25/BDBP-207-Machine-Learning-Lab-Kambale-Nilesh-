#2.​ Compute the dot product of two vectors, x and y given below TT
#x = [2 1 2]T and y = [1 2 2]T . What is the meaning of the dot product of two vectors?
#Illustrate that with your own example.
x=[[2,1,2]]
y=[[1,2,2]]

def tranpose_matrix(a):
    result=[]
    for i in range(len(a[0])):
        row=[]
        for j in range(len(a)):
            row.append(a[j][i])

        result.append(row)
    return result
xt=tranpose_matrix(x)
yt=tranpose_matrix(y)
#print(xt)
#print(yt)

def dot(a,b):
    result=0
    for i in range(len(a)):
        for j in range(len(b[0])):
            result+=a[i][j]*b[i][j]
    return result
dot_product=dot(xt,yt)
print(f'Dot product of transpose matrix ofx & y = {dot_product}')
            

            
