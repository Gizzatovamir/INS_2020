import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
from keras.models import Sequential
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
print(training_data)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
#print("Categories:", np.unique(targets))
#print("Number of unique words:", len(np.unique(np.hstack(data))))
length = [len(i) for i in data]
#print("Average Review length:", np.mean(length))
#print("Standard Deviation:", round(np.std(length)))
#print("Label:", targets[0])
#print(data[0])
index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]] )
#print(decoded)

user_txt = [
    "The movie affects you in a way that makes it physically painful to experience, but in a good way",
    "Seriously, unbelievable",
    "One of the best movies in these few years",
    "It is very boring film",
    "it's very good",
    "it's very boring",
    "fantastic film, wonderful casting, good job, creators",
    "beautiful picture, good scenario, it's amazing",
    "Absolute disappointment. I believed the hype like a fool"
]
user_y = [1., 1., 1.,0., 1., 0., 1., 1., 0.]

def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def index(str, indx):
    lst = str.split()
    for i, w in enumerate(lst):
        lst[i] = indx.get(w)
    return lst


def user_text(str,indx):
    for i in range(len(str)):
        str[i] = index(str[i],indx)
    return str


def set_zero(str):
    for i,j in enumerate(str):
        for k,value in enumerate(j):
            if value is None:
                str[i][k] = 0
    return str

user_x = user_text(user_txt,imdb.get_word_index())
set_zero(user_txt)

data = vectorize(data)
targets = np.array(targets).astype("float32")
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]
#print(train_x[:1000])
# Input - Layer
model = Sequential()
model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))
# Hidden - Layers
model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
model.add(layers.Dense(100, activation = "relu"))
model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()
model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"])
H = model.fit(
 train_x, train_y,
 epochs= 2,
 batch_size = 50,
 validation_data = (test_x, test_y)
)
loss = H.history['loss']
v_loss = H.history['val_loss']
plt.plot(loss, 'b', label='train')
plt.plot(v_loss, 'r', label='validation')
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.clf()

acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
plt.plot(acc, 'b', label='train')
plt.plot(val_acc, 'r', label='validation')
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.clf()

user_x = vectorize(user_x)

user_loss, user_acc = model.evaluate(user_x, user_y)

print('user_acc:', user_acc)
preds = model.predict(user_x)
plt.title("user dataset predications")
plt.plot(user_y, 'r', marker='v', label='truth')
plt.plot(preds, 'b', marker='x', label='pred')
plt.legend()
plt.show()
plt.clf()