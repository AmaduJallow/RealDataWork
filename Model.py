import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def predictor(year, weight, bias):
    return year * weight + bias


def loss_function(year, population, weight, bias):
    return np.average((predictor(year, weight, bias) - population) ** 2)


def gradient_function(year, populaion, weight, bias):
    weight_gradient = 2 * np.average(year * (predictor(year, weight, bias) - populaion))
    bias_gradient = 2 * np.average(predictor(year, weight, bias) - populaion)
    return weight_gradient, bias_gradient


def train_model(year, population, interations, theta):
    weight = bias = 0
    for i in range(interations):
        weight_gradient, bias_gradient = gradient_function(year, population, weight, bias)
        weight -= weight_gradient * theta
        bias -= bias_gradient * theta
    return weight, bias


director = "C:/Users/amadu/PycharmProjects/RealDataWork/Data.csv"

year, population = np.loadtxt(director, delimiter=",", unpack=True, dtype="float64")
plt.plot(year * 1000, population * 1000, "bo")
weight, bias = train_model(year, population, 1_000_000, 0.2011)
print(weight, bias)
print(f"The population in the year 2023 is {predictor(2.023, weight, bias) * 1000}")

newweight = 34002.27681327201
newbias = -66679.07308859545


def predictor_xx(year):
    print(f"The population for the year {int(year * 1000)} is {1000*((year * newweight) + newbias)}")


predictor_xx(2.023)
plt.show()
