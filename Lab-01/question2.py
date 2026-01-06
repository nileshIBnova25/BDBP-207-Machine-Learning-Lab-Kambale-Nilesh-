# Implement y = 2x1 + 3 and pmat plotlot x1, y [start=-100, stop=100, num=100]
import matplotlib.pyplot as plt
x1=[ i for i in range(-100,100,2) ]
def cpl(x):
    return (2*x)+3

y=list(map(cpl,x1))

plt.plot(x1, y, label='y = 2x + 3')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Function Plot')
plt.legend()
plt.grid(True)
plt.show()
