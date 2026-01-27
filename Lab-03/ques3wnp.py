x = [[1,1,2],[1,2,1],[1,3,3]]
y = [3,4,5]
theta = [0,0,0]
alpha = 0.001


def hypothesis_fun(x, theta, s):
    total = 0
    for i in range(len(theta)):
        total += x[s][i] * theta[i]
    return total


def deriv_fun(x, y, theta, t):
    z = 0
    for i in range(len(x)):
        z += (hypothesis_fun(x, theta, i) - y[i]) * x[i][t]
    return z



def main():
     
    def cost_fun(theta, x, y, alpha):

        for itr in range(1000):
            new_theta = [0] * len(theta)

            for f in range(len(theta)):
                new_theta[f] = theta[f] - alpha * deriv_fun(x, y, theta, f)

                if (new_theta[0] - theta[0]) <= 0.001  :
                    return theta

            theta = new_theta  # update theta correctly
            print(f"Iteration {itr}: theta = {theta}")

        return theta



    result = cost_fun(theta, x, y, alpha)
    print("Final theta:", result)

if __name__ == "__main__":
    main()