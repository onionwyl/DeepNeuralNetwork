import dnn
import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime
from PIL import Image
from scipy import ndimage, misc

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
if __name__ == '__main__':
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    # reshape data to [image_vector, data_set_num] and
    train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T / 255.
    test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T / 255.
    input_layer_num = train_x.shape[0]
    train_y = np.row_stack((train_y, np.array(train_y == 0).astype(int)))
    test_y = np.row_stack((test_y, np.array(test_y == 0).astype(int)))
    # try a 5-layer model
    layer_dims = [input_layer_num, 20, 7, 5, 2]
    activation_dims = ["relu", "relu", "relu", "relu","softmax"]
    learning_rate = 0.005
    num_iterations = 4000
    costs = []
    parameters = dnn.initialize_parameters(layer_dims)
    for i in range(0, num_iterations):
        A_pred, caches = dnn.forward_propagation(train_x, parameters, activation_dims)
        cost = dnn.compute_cost(A_pred, train_y, out_activation=activation_dims[-1])
        grads = dnn.backward_propagation(A_pred, train_y, caches, activation_dims)
        parameters = dnn.update_parameters(parameters, grads, learning_rate)
        print(len(parameters))
        # print(cost)
        if i % 100 == 0:
            print("Iteration: %i Cost: %f" % (i, cost))
            costs.append(cost)
    np.save('parameters_cat_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), parameters)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('Cost graph with learning rate' + str(learning_rate))
    plt.show()