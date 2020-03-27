import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)
print(test_data,'test_data')
print(train_data,'train_data')
print(test_targets)
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std




def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

k = 5
num_val_samples = len(train_data) // k
num_epochs = 80
all_scores = []
res = []
def test():
    for i in range(k):
        print('processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
        model = build_model()
        a = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1,verbose=0, validation_data=(val_data, val_targets))
        loss = a.history['loss']
        mae = a.history['mean_absolute_error']
        v_loss = a.history['val_loss']
        v_mae = a.history['val_mean_absolute_error']
        x = range(1, num_epochs + 1)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
        plt.plot(x, loss)
        plt.plot(x, v_loss)
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(['Train data', 'Test data'], loc='upper left')
        plt.show()
        plt.plot(x, mae)
        plt.plot(x, v_mae)
        plt.title('Model mean absolute error')
        plt.ylabel('mean absolute error')
        plt.xlabel('epochs')
        plt.legend(['Train data', 'Test data'], loc='upper left')
        plt.show()

def find_k():
    res = []
    for i in range(k):
        print('processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]],
            axis=0)
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1,
                            verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
        res.append(np.mean(all_scores))
    plt.plot(range(k), res)
    plt.title('Dependence on k')
    plt.ylabel('Mean')
    plt.xlabel('k')
    plt.show()
    print(np.mean(all_scores))


test()
#find_k()
print(np.mean(all_scores))