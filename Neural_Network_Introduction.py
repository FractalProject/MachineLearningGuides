import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Extract the training and test data
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images  # Returns np.array
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# Randomly shuffle the data
train_idx = np.random.permutation(train_data.shape[0])
test_idx = np.random.permutation(test_data.shape[0])
train_data = train_data[train_idx]
train_labels = train_labels[train_idx]
test_data = test_data[test_idx]
test_labels = test_labels[test_idx]

# Familiarize yourself with some information about the data
print(type(train_data))
print(type(train_labels))
print(train_data.shape)
print(train_labels.shape)
print("\n")
print("First 10 entires: \n" + str(train_data[0:10]))
print("First 10 labels: \n" + str(train_labels[0:10]))

# Visualize the data
example = np.reshape(train_data[0], (28,28))
plt.imshow(example)