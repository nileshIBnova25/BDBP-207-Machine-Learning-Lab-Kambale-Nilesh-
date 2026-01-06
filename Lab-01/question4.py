#Implement Gaussian PDF - mean = 0, sigma = 15 in the range[start=-100, stop=100, num=100]
import matplotlib.pyplot as plt
import math
start=-100
stop=100
count=100
step=(stop-start)/count
x=[]
for i in range(count):
    value = start + (i * step)
    x.append(value)

mean=0
sigma=15
def pdff(x):
    return (1/(sigma*math.sqrt(2*math.pi))) * math.exp(-0.5*((x-mean)/sigma)**2)
pdf=list(map(pdff,x))


plt.plot(x,pdf,label='gaussian PDF')
plt.xlabel('x')
plt.ylabel('probability density')
plt.title('gaussian (normal) distribution')
plt.legend()
plt.grid(True)
plt.show()

