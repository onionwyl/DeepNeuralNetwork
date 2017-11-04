import numpy as np
import gzip
import matplotlib.pyplot as plt

# Load datasets from mnist gz file
# The code is written after I read tensorflow's mnist.py
# Image file :
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  60000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel
#
# Label file:
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  10000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label
#
# For more information, you can visit the following website:
# Download mnist and read introductions: http://yann.lecun.com/exdb/mnist/
# Tensorflow mnist code: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/learn/python/learn/datasets/mnist.py

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')            # A new dtype with big endian order
    return np.frombuffer(bytestream.read(4), dtype=dt)[0] # Read 4 bytes from file into the new type(32bit int)

def extract_label(path):
    with gzip.open(path, 'rb') as bytestream:
        magic = _read32(bytestream)                     # Check the magic number(if necessary)
        num_labels = _read32(bytestream)                # Read next 4 bytes: num of labels
        buf = bytestream.read(num_labels)               # Read labels with each label a 8 bit uint
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_labels, 1)              # Reshape data to a [num_labels, 1] numpy array
        return data

def extract_image(path):
    with gzip.open(path, 'rb') as bytestream:           # Same as above
        magic = _read32(bytestream)
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows*cols*num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]                            # Make labels from dense to one hot:
    index_offset = np.arange(num_labels) * num_classes            # [0, 1, 2] => [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1  # Set all labels in one hot array
    return labels_one_hot                                         # For more information see numpy.ndarray.flat, numpy.ravel

def load_training_set():
    MNIST_PATH = 'mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    train_images = extract_image(MNIST_PATH+TRAIN_IMAGES)
    train_labels = extract_label(MNIST_PATH+TRAIN_LABELS)
    test_images = extract_image(MNIST_PATH+TEST_IMAGES)
    test_labels = extract_label(MNIST_PATH+TEST_LABELS)
    return train_images, train_labels, test_images, test_labels
if __name__ == '__main__':
    label = 11
    train_images, train_labels, test_images, test_labels= load_training_set()
    plt.imshow(train_images[label].reshape((28, 28)), cmap='gray')
    plt.title("Label:"+str(train_labels[label]))
    plt.show()