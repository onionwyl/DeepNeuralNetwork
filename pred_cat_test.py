import numpy as np
import dnn
from pred_cat import load_data
import matplotlib.pyplot as plt

def get_accuracy(pred, y):
    same = 0
    for i in range(0, len(pred)):
        # print(pred[i], y[i])
        if np.equal(pred[i], y[i]).all():
            same += 1
        else:
            plt.title("Prediction:"+str(pred[i])+"\nReal:"+str(y[i]))
            plt.imshow(test_x[:, i].reshape(64, 64, 3), interpolation='nearest')
            plt.show()

    return same / len(pred)
if __name__ == '__main__':
    parameters = np.load("parameters_cat_2017-11-04_19-49-13.npy").all()
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    # reshape data to [image_vector, data_set_num] and
    train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T / 255.
    test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T / 255.
    input_layer_num = train_x.shape[0]
    train_y = np.row_stack((train_y, np.array(train_y == 0).astype(int)))
    test_y = np.row_stack((test_y, np.array(test_y == 0).astype(int)))
    # try a 5-layer model
    activation_dims = ["relu", "relu", "relu", "relu","softmax"]
    train_pred, caches = dnn.forward_propagation(train_x, parameters, activation_dims)
    print("Training set Accuracy:"+str(get_accuracy(np.array(train_pred > 0.5).astype(int).T, train_y.T)))
    test_pred, caches = dnn.forward_propagation(test_x, parameters, activation_dims)
    print("Test set Accuracy:"+str(get_accuracy(np.array(test_pred > 0.5).astype(int).T, test_y.T)))
