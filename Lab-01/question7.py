x=[[1,0,2],[0,1,1],[2,1,0],[1,1,1],[0,2,1]]
theta=[[2],[3],[3]]

def matrixmulti(a,b):
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
    print(result)

matrixmulti(x,theta)

