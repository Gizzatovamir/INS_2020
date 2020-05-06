import var3
from keras import *
from keras.layers import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
input_shape = (28, 28, 1)

acc_in_all = []
i = 0
on_epochs = [1, 4, 6, 8, 9, 10, 12]
def add_batch(x):
    acc_in_all.append(x)

class AccuracyHistogramsBuilding(tf.keras.callbacks.Callback):
    def __init__(self, on_epochs):
        super(AccuracyHistogramsBuilding, self).__init__()
        self.on_epochs = on_epochs
        self.accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.on_epochs:
            self.accuracy.append((logs.get('acc'), logs.get('val_acc')))

    def on_train_end(self, logs=None):
        print(self.accuracy)
        plt.bar(list(map(lambda x: x + 0.85, self.on_epochs)), list(map(lambda x: x[0], self.accuracy)), width=0.5,
                label='training acc')
        plt.bar(list(map(lambda x: x + 1.15, self.on_epochs)), list(map(lambda x: x[1], self.accuracy)), width=0.5,
                label='validation acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(prop={'size': 10})
        plt.title('Training and validation accuracy')
        plt.savefig("acc_hist")
        plt.show()


class ModelsSaving(tf.keras.callbacks.Callback):
    def __init__(self, on_epochs):
        super(ModelsSaving, self).__init__()
        self.on_epochs = on_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.on_epochs:
            self.model.save(str(epoch+1) + ".h5")


def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(60, kernel_size=(3,3),activation="relu",input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(60, kernel_size=(3,3),activation="relu"))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation='softmax'))
    return model


def generate_data():
    X, Y = var3.gen_data(img_size=28)
    X = np.asarray(X)
    Y = np.asarray(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train /= 255
    x_test /= 255
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_train = to_categorical(y_train, 2)
    encoder.fit(y_test)
    y_test = encoder.transform(y_test)
    y_test = to_categorical(y_test, 2)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = generate_data()
#print(x_train.shape)
#print(y_train.shape)
#print(x_test)
model = create_model(input_shape)
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['acc'])
H = model.fit(x_train, y_train, batch_size=20, epochs=15, validation_data=[x_test, y_test], callbacks=[AccuracyHistogramsBuilding(list(map(lambda x: x-1, on_epochs))),ModelsSaving(list(map(lambda x: x-1, on_epochs)))])
#print(acc_in_all)


loss = H.history['loss']
v_loss = H.history["val_loss"]
plt.figure(1, figsize=(8, 5))
plt.plot(loss, 'b', label='train')
plt.plot(v_loss, 'r', label='validation')
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.clf()

acc = H.history['acc']
val_acc = H.history['val_acc']
plt.plot(acc, 'b', label='train')
plt.plot(val_acc, 'r', label='validation')
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.clf()


