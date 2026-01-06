#Implement y = 2x12 + 3x1 + 4 and plot x1, y in the range [start=--10, stop=10, num=100]
import matplotlib.pyplot as plt
import random
start=-10
stop=10
count=100
x1 = []
step=(stop-start)/count
for i in range(count):
    value = start + (i * step)
    x1.append(value)

def cpl(x):
    return 2 * x**2 + 3 * x + 4
y=list(map(cpl,x1))

print(x1)

#y=list(map(cpl,x1))
plt.plot(x1,y,label='y=2x^2+3*x+4')
plt.xlabel('x1')
plt.ylabel('y')
plt.title('Quadratic Function Plot')
plt.legend()
plt.grid(True)
plt.show()