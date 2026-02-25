import gzip
import numpy as np
def load_mnist_images(filename):
   with gzip.open(filename, 'rb') as f:
       f.read(16)
       data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
   return data
def load_mnist_labels(filename):
   with gzip.open(filename, 'rb') as f:
       f.read(8)
       labels = np.frombuffer(f.read(), dtype=np.uint8)
   return labels


x_train = load_mnist_images('../data/MNIST/train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('../data/MNIST/train-labels-idx1-ubyte.gz')
x_test = load_mnist_images('../data/MNIST/t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('../data/MNIST/t10k-labels-idx1-ubyte.gz')