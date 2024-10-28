# Handwritten-Digit-Recognition
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the training and test data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=3)

# Save the model
model.save('handwritten.model')

# Load the saved model
model = tf.keras.models.load_model('handwritten.model')

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Start processing the images
image_number = 1
while os.path.isfile(f"digitFolder/digit{image_number}.png"):
    try:
        # Load the image in grayscale
        img = cv2.imread(f"digitFolder/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)

        # Resize the image to 28x28 if necessary (in case it's not the correct size)
      
        img = cv2.resize(img, (28, 28))

        # Invert and normalize the image
        img = np.invert(img)  # Invert the colors (MNIST has white digits on black background)
        img = img / 255.0  # Normalize the pixel values between 0 and 1

        # Reshape the image to (1, 28, 28) to match the input shape the model expects
        img = np.reshape(img, (1, 28, 28))

        # Predict the digit
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")

        # Display the image
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        image_number += 1
