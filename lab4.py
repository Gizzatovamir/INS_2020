import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tensorflow.keras

def get_image(path):
    img = image.load_img(path=path,grayscale=True, target_size=(28, 28, 1))
    result = image.img_to_array(img)
    return result


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
plt.imshow(train_images[500], cmap=plt.cm.binary)
plt.show()
print(train_images[130].shape)
#print(train_images.shape)
#print(train_labels)
#print(train_labels[130])
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#plt.imshow(train_images[800], cmap=plt.cm.binary)
#plt.show()
model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
adam = tensorflow.keras.optimizers.Adam(learning_rate=0.02)
Rmsprop = tensorflow.keras.optimizers.RMSprop(learning_rate=0.0002)
sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)
adagard = tensorflow.keras.optimizers.Adagrad(learning_rate=0.002)
model.compile(optimizer=adagard, loss='categorical_crossentropy', metrics=['accuracy'])
H = model.fit(train_images, train_labels, epochs=6, batch_size=128, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print(H.history.keys())


loss = H.history['loss']
val_loss = H.history['val_loss']
x = range(1, 11)
#plt.plot(x, loss, 'b', label='loss')
#plt.plot(x, val_loss, 'r', label='val_loss')
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
#plt.show()
plt.clf()

acc = H.history['acc']
val_acc = H.history['val_acc']
#x = range(1, 11)
#plt.plot(x, acc, 'b', label='acc')
#plt.plot(x, val_acc, 'r', label='val_acc')
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend()
#plt.show()
plt.clf()


test = get_image('./Untitled.png')
test /= 255.0
plt.imshow(1 - test.reshape(28,28), cmap=plt.cm.binary)
#plt.show()
#print(abs(test[:, :, 0]-1))
#model.predict(test[:, :, 0])
#test_img = abs(1 - test[:, :, 0]).reshape((1, 784))
img_class = model.predict_classes(1 - test.reshape((1, 784)))
#print("Test class: ", img_class)

test_7 = get_image('./7_2.png')
test_7 /= 255.0
plt.imshow(1 - test_7.reshape((1,784)), cmap=plt.cm.binary)
#plt.show()
#print(abs(test_7[:, :, 0]-1))
#model.predict(test[:, :, 0])
test_7_img = test_7.reshape((1, 28, 28))
img_7_class = model.predict_classes(1 - test_7_img)
#print("Test_7 class: ", img_7_class)


test_9 = get_image('./9.png')
test_9 /= 255.0
plt.imshow(1 - test_9.reshape((1, 784)), cmap=plt.cm.binary)
#plt.show()
#print(abs(test_7[:, :, 0]-1))
#model.predict(test[:, :, 0])
test_9_img = test_9.reshape((1, 28, 28))
img_9_class = model.predict_classes(1 - test_9_img)
#print("Test_9 class: ", img_9_class)

img_1_class = model.predict_classes(train_images[500].reshape((1,784)))
print("Test_9 class: ", img_1_class)