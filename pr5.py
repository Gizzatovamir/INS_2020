import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from tensorflow.keras.models import load_model
from cmath import pi
import csv
import collections
import matplotlib.pyplot as plt
col = 6
x_sigma = np.sqrt(10)
x_mu = 0
y_sigma = np.sqrt(0.3)
y_mu = 0
train_size = 400

def genData(size):
    data = np.zeros((size, col))
    data_y = np.zeros(size)
    for i in range(size):
        x = x_sigma * np.random.randn(1) + x_mu
        e = y_sigma * np.random.randn(1) + y_mu
        data[i,:] = (x**2 + x + e, abs(x) + e, np.sin(x - pi/4)+e, -x**3+e, -x/4 + e, -x + e)
        data_y[i] = np.log10(abs(x)) + e
    mean_x = np.mean(data, axis=0)
    std_x = np.std(data, axis=0)
    data = (data - mean_x)/std_x
    mean_y = np.mean(data_y, axis=0)
    std_y = np.std(data_y, axis=0)
    data_y = (data_y - mean_y)/std_y
    return np.array(np.round(data, 3)), np.array(np.round(data_y, 3))


def save_csv(name, data):
    file = open(name, "w+")
    my_csv = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    if isinstance(data, collections.Iterable) and isinstance(data[0], collections.Iterable):
        for i in data:
            my_csv.writerow(i)
    else:
        my_csv.writerow(data)


encode_input = Input(shape=(col,), name="encode_input")
ecd = Dense(15, activation="relu")(encode_input)
ecd = Dense(30, activation="relu")(ecd)
ecd = Dense(30, activation="relu")(ecd)
ecd = Dense(col, activation="linear")(ecd)

dec_input = Input(shape=(col,), name="input_decoded")
decd = Dense(15, activation="relu")(dec_input)
decd = Dense(30, activation="relu")(decd)
decd = Dense(30, activation="relu")(decd)
decd = Dense(col, name="auto_aux")(decd)

predicted = Dense(15, activation="relu", kernel_initializer="normal", name="predict")(ecd)
#predicted = Dense(60, activation="relu")(predicted)
predicted = Dense(30, activation="relu")(predicted)
predicted = Dense(30, activation="relu")(predicted)
predicted = Dense(1, name="main_out")(predicted)

x_train , y_train = genData(train_size)
x_test , y_test = genData(train_size)
encoder = Model(encode_input, ecd, name="encoder")
decoder = Model(dec_input, decd, name="decoder")
predicter = Model(encode_input, predicted, name="predicter")
predicter.compile(optimizer="adam", loss="mse", metrics=["mae"])
#a = predicter.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=[x_test, y_test])
a = predicter.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2)

model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse",metrics=['mae'])
H = model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2)

#loss comparison
loss = a.history['loss']
model_loss = H.history['loss']
x = range(0, 100)
plt.plot(x, loss, 'b', label='my_loss')
plt.plot(x, model_loss, 'r', label='model_loss')
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.clf()
# val_loss comparison
my_train = a.history['val_loss']
model_train = H.history['val_loss']
plt.plot(x, my_train, 'b', label='my_val_loss')
plt.plot(x, model_train, 'r', label='model_val_loss')
plt.title('val_loss')
plt.ylabel('val_loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.clf()

decoder.save('decoder.h5')
encoder.save('encoder.h5')
predicter.save('predicter.h5')



save_csv('./x_train.csv', x_train)
save_csv('./y_train.csv', y_train)
save_csv('./x_test.csv', x_test)
save_csv('./y_test.csv', y_test)
save_csv('./encoded.csv', encoder.predict(x_test))
save_csv('./decoded.csv', decoder.predict(encoder.predict(x_test)))
save_csv('./regr.csv', predicter.predict(x_test))


#print(genData(100))
#save_csv("./test.csv",genData(train_size)[0])
