import numpy as np
from keras import Sequential
from keras.layers import Dense


def log_fun(a, b, c):
    return (a and b) or c


def res(mat):
    return np.array([log_fun(*i) for i in mat])


def sigmoid(x):
    return 1/(1+np.exp(-x))


def my_relu(x):
    return np.maximum(x, 0)


def through_tensor(data, weights):
    res = data.copy()
    layers = []
    for i in range(len(weights)-1):
        layers.append(my_relu)
    layers.append(sigmoid)
    for i in range(len(weights)):
        res = layers[i](np.dot(res, weights[i][0])+weights[i][1])
    return res


def poelem(data, weights):
    res = data.copy()
    layers = []
    for i in range(len(weights)-1):
        layers.append(my_relu)
    layers.append(sigmoid)
    for i in range(len(weights)):
        step = np.zeros((len(res), len(weights[i][1])))
        for j in range(len(res)):
            for k in range(len(weights[i][1])):
                sum =0
                for l in range(len(res[j])):
                    sum += res[j][l]*weights[i][0][l][k]
                step[j][k] = layers[i](sum + weights[i][1][k])
        res = step
    return res


def test(model, data):
    weights = [i.get_weights() for i in model.layers]
    t_res = through_tensor(data, weights)
    each = poelem(data, weights)
    m_res = model.predict(data)
    print("tensor")
    print(t_res)
    print("model")
    print(m_res)
    print("poelem")
    print(each)


if __name__ == '__main__':
    a = np.genfromtxt('pr4.txt', delimiter=';')
    print(a)
    model = Sequential()
    model.add(Dense(8, activation="relu", input_shape=(3,)))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    print("before")
    test(model, a)
    model.fit(a, res(a), epochs=20, batch_size=1)
    print("after")
    test(model, a)
    print("predict")

    #print(model.predict(a))
