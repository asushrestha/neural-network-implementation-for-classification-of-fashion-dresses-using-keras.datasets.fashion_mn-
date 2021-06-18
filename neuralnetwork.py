import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),  # number of hidden neurons
    keras.layers.Dense(10, activation="softmax")  # activation function used is softmax
])
'''creating the neural network model in sequence of input hidden and output alyer'''

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# search for adam optimizer and sparse_categorical_crossentropy for math behind this.
model.fit(train_images, train_labels, epochs=5)  # number of time we feed system the image.
'''greater epoch doesn't guarantee better model try 5,10 epochs to test'''

# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print("Testing accuracy:", test_acc)
'''use this to check accuracy'''

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual:" + class_names[(test_labels[i])])
    plt.title("Prediction " + class_names[np.argmax(prediction[i])])
    plt.show()
# display the images with actual image data and predicted image data.
# print(class_names[np.argmax(prediction[0])])