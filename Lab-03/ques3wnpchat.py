x = [[1,1,2],[1,2,1],[1,3,3]]
y = [3,4,5]
theta = [0,0,0]
alpha = 1e-2
r=1000






def hypothesis_fun(x, theta, i2):
    total = 0
    for i1 in range(len(theta)):
        total += x[i2][i1] * theta[i1]
    return round(total,5)


def deriv_fun(x, y, theta, i3):
    z = 0
    for i2 in range(len(x)):
        z += (hypothesis_fun(x, theta, i2) - y[i2]) * x[i2][i3]
    return z



def main():

    def get_theta(theta, x, y, alpha,r):

        for itr in range(0,r):
            new_theta = [0] * len(theta)

            for i3 in range(len(theta)):
                new_theta[i3] = theta[i3] - alpha * deriv_fun(x, y, theta, i3)

                if (new_theta[0] - theta[0]) <= 0.01  :
                    return theta

            theta = new_theta  # update theta correctly
            print(f"Iteration {itr}: theta = {theta}")

        return theta



    result = get_theta(theta, x, y, alpha,r)
    print("Final theta:", result)

if __name__ == "__main__":
    main()
