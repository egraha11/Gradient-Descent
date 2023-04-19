import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


x = 2 * np.random.rand(100)
y = 3 * x + np.random.rand(100)


#define a cost function for gradient descent to use
def RSS(y_pred, y):

    return (np.sum(np.square(y_pred - y)))


def gradient_descent(x, y, learning_rate = .0001, max_iterations = 1000, min_step = .1):

    current_cost = sys.maxsize
    iteration = 0

    archived_costs = []
    archived_weights = []
    archived_bias = []

    #initialize intercept and slope 
    weight = np.random.rand()
    bias = np.random.rand()

    archived_costs.append(current_cost)

    preds = []

    while iteration < max_iterations:

        #make predictions
        preds = weight * x + bias

        #calculate error with chosen cost function 
        current_cost = RSS(preds, y)

        #save previous values
        archived_costs.append(current_cost)
        archived_weights.append(weight)
        archived_bias.append(bias)

        if abs(current_cost - archived_costs[-2]) < min_step:
            break

        #take the partial derivitives with respect to each factor

        weight_deriv = -2 * sum(y - preds)
        bias_deriv = -2 * sum(x * (y - preds))


        #calculate the new slope and intercept
        weight = weight - (weight_deriv * learning_rate)
        bias = bias - (bias_deriv * learning_rate)

        iteration += 1

        print(f"Iteration: {iteration}")
        print(f"Cost: {current_cost}")
        print(f"Weight: {weight}")
        print(f"Bias: {bias} \n --------")

    archived_costs.pop(0)

    plt.plot(archived_weights, archived_costs, color="red")
    plt.scatter(archived_weights, archived_costs, color="blue")
    plt.xlabel("Weight Variable")
    plt.ylabel("Cost Variable")
    plt.title("Costs vs Weights")
    plt.show()


    plt.scatter(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot([min(x), max(x)], [min(y), max(y)])
    plt.show()

    return weight, bias, archived_costs, archived_bias, archived_weights




gradient_descent(x, y)

