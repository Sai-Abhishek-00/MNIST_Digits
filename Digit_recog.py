import random
import matplotlib.pyplot as plt
import tensorflow as tf
# Read & split the dataset into test and train
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from matplotlib import pyplot

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

#¤ print shape of the data set and see the split
print("x train shape = {0}" .format(x_train.shape))
print("y train shape = {0}" .format(y_train.shape))
print("x test shape = {0}" .format(x_test.shape))
print("y test shape = {0}" .format(y_test.shape))


# print images
for i in range(0,5):
    image_index = random.randint(0,59999)
    print(y_train[image_index])
    plt.imshow(x_train[image_index])
    plt.show()




#normalize images in training data set. for model evaluation divide training data into train and test

#convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#reshape into 4 dimensions for KERAS API compatibility
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

#divide by 255 to normalize
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape)) #32 filters, 3*3 frame of filter
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu)) #dense layer with 128 nodes for feature extraction
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax)) #10 node output layer, one for each output classification label

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10)

model.evaluate(x_test, y_test)

# save model
model.save('digit_recog_model.h5')