from numpy import *


def compute_error_for_line(b, m, data):
    # init error at 0
    error = 0
    for i, val in enumerate(data):
        x = val[0]
        y = val[1]
        # apply formula
        error += (y - (m * x + b)) ** 2
    return error / float(len(data))


def get_gradient_step(cur_b, cur_m, data, learning_rate):
    b_grad = 0
    m_grad = 0
    n = float(len(data))

    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        b_grad += -(2 / n) * (y - (cur_m * x) + cur_b)
        m_grad += -(2 / n) * x * (y - (cur_m * x) + cur_b)
    new_b = cur_b - (learning_rate * b_grad)
    new_m = cur_m - (learning_rate * m_grad)
    return [new_b, new_m]


def get_gradient_descent(data, initial_b, initial_m, learning_rate, nr_iterations):
    b = initial_b
    m = initial_m
    for i in range(nr_iterations):
        b, m = get_gradient_step(b, m, data, learning_rate)
    return [b, m]


def run():
    # colect data
    data = genfromtxt('data.csv', delimiter=',')

    # defign hyper parameters
    learning_rate = 0.0001
    # y = mx + b (slope formula)
    initial_m = 0
    initial_b = 0
    nr_iterations = 1000
    # initial training
    print(
        "Starting gradient descent at b = {}, m= {}, error {}"
            .format(initial_b, initial_m, compute_error_for_line(initial_m, initial_b, data))
    )
    [b, m] = get_gradient_descent(data, initial_b, initial_m, learning_rate, nr_iterations)

    print(
        "Ending Point at b = {}, m= {}, error {}"
            .format(b, m, compute_error_for_line(b, m, data))
    )


if __name__ == '__main__':
    run()
