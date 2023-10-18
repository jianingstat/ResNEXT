import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:00.
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    x_train, y_train=np.empty(shape=(0,3072)),np.array([])
    for i in range(5):
        file = data_dir+"/data_batch_"+str(i+1)
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        x_train = np.vstack([x_train, np.array(dict[b'data'])])
        y_train = np.hstack([y_train, np.array(dict[b'labels'])])
    file = data_dir+"/test_batch"
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    x_test = np.array(dict[b'data'])
    y_test = np.array(dict[b'labels'])
    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    testing_file = os.path.join(data_dir, 'private_test_images_2022.npy')
    with open(testing_file, 'rb') as f:
        d = np.load(f, encoding='latin1')
    x_test = d.astype(np.float32).reshape((-1, 3072))
    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    indices = np.random.permutation(x_train.shape[0])
    n_train = int(x_train.shape[0]*train_ratio)
    train_idx, valid_idx = indices[:n_train], indices[n_train:]
    x_train_new = x_train[train_idx]
    y_train_new = y_train[train_idx]
    x_valid = x_train[valid_idx]
    y_valid = y_train[valid_idx]
    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

