# Implement y = x1^2, plot x1, y in the range [start=--10, stop=10, num=100].
# Compute the value of derivatives at these points, x1 = -5, -3, 0, 3, 5.
# What is the value of x1 at which the function value (y) is zero. What do you infer from this
import math

import matplotlib.pyplot as plt
start=-10
stop=10
count=100
step=(stop-start)/count
x1=[]
for i in range(count):
    value = start + (i * step)
    x1.append(value)
def yval(x):
    return x*x
y=list(map(yval,x1))



print(y)
plt.plot(x1,y,label='y=x**2')
plt.xlabel('x1')
plt.ylabel('y')
plt.title('Plot of y = x2')
plt.legend()
plt.grid(True)


d=[-5,-3,0,3,5]
diff=list(map(lambda x :x*2,d))
plt.plot(d,diff,label='d=x**2')
plt.show()
for i in range(len(d)):
    print(f"at point x1={d[i]} derivative={diff[i]}")



#print(deri)
#for x,a in zip(d,deri):
#    print(f"Derivative at x1 = {x} : {a}")



