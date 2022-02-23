import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import cv2
import os



def preprocessData(X_train, X_test):

    # normalize data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # add one dimension
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, X_test


def setModel():

    my_input = keras.layers.Input(shape=(28, 28, 1))

    x = (keras.layers.Conv2D(16, kernel_size=3, padding='same',
                             activation='relu', input_shape=(28, 28, 1)))(my_input)
    x = (keras.layers.MaxPooling2D())(x)

    x = (keras.layers.Conv2D(64, kernel_size=3, 
                             padding='same', activation='relu'))(x)
    x = (keras.layers.MaxPooling2D())(x)

    x = (keras.layers.Conv2D(64, kernel_size=3, 
                             padding='same', activation='relu'))(x)
    x = (keras.layers.MaxPooling2D())(x)
    x = (keras.layers.Dropout(0.25))(x)

    x = (keras.layers.Flatten())(x)
    x = (keras.layers.Dense(units=128, activation='relu'))(x)
    x = (keras.layers.Dense(units=10, activation='softmax'))(x)

    model = keras.Model(inputs=my_input, outputs=x)

    return model


def main():

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    X_train, X_test = preprocessData(X_train, X_test)

    model = setModel()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

    test_accuracy = model.evaluate(X_test, y_test)

# Let's try to guess the numbers that we wrote down in our own handwriting.
# I used paint for this

    image_number = 0
    images = []
    labels = []
    plt.figure(figsize=(12, 8))

    while os.path.isfile(f'numbers/digit{image_number}.png'):
        try:

            image = cv2.imread(f'numbers/digit{image_number}.png', 0)
            image = np.invert(np.array([image]))
            pred_image = model.predict(image)
            images.append(image)
            labels.append(str(np.argmax(pred_image)))

        finally:
            image_number += 1

    for i in range(10):

        plt.subplot(2, 5, i+1)

        plt.title(f'That is probably {labels[i]}', size=13)
        plt.imshow(images[i][0], cmap=plt.cm.binary)

    plt.show()

    
if __name__ == '__main__':
    main()
