# Import dataset, from another files, already scaled (StandardScaler) and splitted
import time

import hpelm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from loadMNIST_orig import X_test, y_test, X_train, y_train


# Useful functions

def correct_prediction(labels, prediction):
    """Reshape vector from one-hot-encode format and create a boolean array where each image is correctly classified"""

    cls_true = labels.argmax(1)
    cls_pred = prediction.argmax(1)
    correct = (cls_true == cls_pred)
    return correct


def ensemble_prediction():
    pred_labels = []
    test_accuracies = []
    train_accuracies = []

    for i in range(n_estimator):
        test_acc = correct_prediction(y_test, y_test_predicted[i])
        test_acc = test_acc.mean()
        test_accuracies.append(test_acc)

        train_acc = correct_prediction(y_train, y_train_predicted[i])
        train_acc = train_acc.mean()
        train_accuracies.append(train_acc)

        msg = "Network: {0}, Accuracy on Training-Set: {1:.6f}, Test-Set: {2:.6f}"
        print(msg.format(i, train_acc, test_acc))

        pred_labels = np.array(y_test_predicted)

    return pred_labels, test_accuracies, train_accuracies


#######################################################################################################################

# HYPERPARAMETERS
np.set_printoptions(precision=2)
np.random.seed(42)

n_estimator = 5

neuron_number = 512
out_class = 10
CV_folds = 10
batch_size = 512
prec = "single"

# Reshape and apply OneHotEncoder to compute the 10class classifier
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

print()
# init list of models and prediction
model = [None] * n_estimator
y_train_predicted = []
y_test_predicted = []
for i in range(n_estimator):
    print('Estim #%d' % i)
    model[i] = hpelm.ELM(X_train.shape[1], out_class, classification="c", batch=batch_size, accelerator="GPU",
                         precision=prec)
    model[i].add_neurons(neuron_number, 'sigm')
    print(str(model[i]))
    t = time.time()
    model[i].train(X_train, y_train, 'c')
    elapsed_time_train = time.time() - t
    pred_train = model[i].predict(X_train)
    y_train_predicted.append(pred_train)
    print("Training time: %f" % elapsed_time_train)
    print('Training Accuracy: ', (1 - model[i].error(y_train, pred_train)))
    print()
    pred_test = model[i].predict(X_test)
    y_test_predicted.append(pred_test)

print("Training done!")
print("###############################################################################################")

# pred labels on test-set
pred_labels, test_accuracies, train_accuracies = ensemble_prediction()

print("\nMean test-set accuracy: {0:.4f}".format(np.mean(test_accuracies)))
print("Min test-set accuracy:  {0:.4f}".format(np.min(test_accuracies)))
print("Max test-set accuracy:  {0:.4f}".format(np.max(test_accuracies)))

# building ensemble
# TODO: hard voting and soft voting
ensemble_pred_labels = np.mean(pred_labels, axis=0)
ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)  # one-hot-reverted
ensemble_correct = (ensemble_cls_pred == y_test.argmax(1))
ensemble_incorrect = np.logical_not(ensemble_correct)

# best network
best_net = np.argmax(test_accuracies)
best_net_pred_labels = pred_labels[best_net, :, :]
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)
best_net_correct = (best_net_cls_pred == y_test.argmax(1))
best_net_incorrect = np.logical_not(best_net_correct)

# Ensemble and Best network comparison
print("\nBest Net correct estimated instances: ", np.sum(best_net_correct))
print("Ensamble correct estimated instances: ", np.sum(ensemble_correct))

ensemble_better = np.logical_and(best_net_incorrect, ensemble_correct)
best_net_better = np.logical_and(best_net_correct, ensemble_incorrect)
print("Best Net better classification: ", best_net_better.sum())
print("Ensemble better classification: ", ensemble_better.sum())

ensemble_acc = correct_prediction(y_test, ensemble_pred_labels)
ensemble_acc = ensemble_acc.mean()
best_net_acc = test_accuracies[best_net]
print("\nEnsemble accuracy: ", ensemble_acc * 100)
print("Best net accuracy: ", test_accuracies[best_net] * 100)

# Accuracy metrics
print("###############################################################################################")
print("Ensemble insight")
class_report_ensemble = classification_report(y_test.argmax(1), ensemble_cls_pred)
cnf_matrix_ensemble = confusion_matrix(y_test.argmax(1), ensemble_cls_pred)
print("Ensemble confusion Matrix:\n", cnf_matrix_ensemble)
print("Ensemble Classification report\n: ", class_report_ensemble)

print("###############################################################################################")
print("Best net insight")
class_report_bestnet = classification_report(y_test.argmax(1), best_net_cls_pred)
cnf_matrix_bestnet = confusion_matrix(y_test.argmax(1), best_net_cls_pred)
print("Best Net confusion Matrix:\n", cnf_matrix_bestnet)
print("Best Net classification report\n: ", class_report_bestnet)
print("###############################################################################################")

#######################################################################################################################
# Helper-functions for plotting and printing comparisons
import matplotlib.pyplot as plt


def plot_images(images,  # Images to plot, 2-d array.
                cls_true,  # True class-no for images.
                ensemble_cls_pred=None,  # Ensemble predicted class-no.
                best_cls_pred=None):  # Best-net predicted class-no.

    assert len(images) == len(cls_true)

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if ensemble_cls_pred is None:
        hspace = 0.3
    else:
        hspace = 1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # For each of the sub-plots.
    for i, ax in enumerate(axes.flat):

        # There may not be enough images for all sub-plots.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i].reshape(img_shape), cmap='binary')

            # Show true and predicted classes.
            if ensemble_cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                msg = "True: {0}\nEnsemble: {1}\nBest Net: {2}"
                xlabel = msg.format(cls_true[i],
                                    ensemble_cls_pred[i],
                                    best_cls_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_images_comparison(idx):
    plot_images(images=X_test[idx, :],
                cls_true=y_test.argmax(1)[idx],
                ensemble_cls_pred=ensemble_cls_pred[idx],
                best_cls_pred=best_net_cls_pred[idx])


def print_labels(labels, idx, num=1):
    # Select the relevant labels based on idx.
    labels = labels[idx, :]

    # Select the first num labels.
    labels = labels[0:num, :]

    # Round numbers to 2 decimal points so they are easier to read.
    labels_rounded = np.round(labels, 2)

    # Print the rounded labels.
    print(labels_rounded)


def print_labels_ensemble(idx, **kwargs):
    print_labels(labels=ensemble_pred_labels, idx=idx, **kwargs)


def print_labels_best_net(idx, **kwargs):
    print_labels(labels=best_net_pred_labels, idx=idx, **kwargs)


def print_labels_all_nets(idx):
    for i in range(num_networks):
        print_labels(labels=pred_labels[i, :, :], idx=idx, num=1)


# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

plot_images_comparison(idx=ensemble_better)
plot_images_comparison(idx=best_net_better)

'''
print('\nELM-GPU single')
elm = hpelm.ELM(X_train.shape[1],out_class, classification ="c", batch=batch_size, accelerator="GPU", precision='single')
elm.add_neurons(neuron_number,'sigm')
#elm.add_neurons(X_train.shape[1],'lin')
print(str(elm))
# Training model
t = time.time()
elm.train(X_train,y_train, 'c')
elapsed_time_train = time.time() - t
y_train_predicted = elm.predict(X_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ',(1-elm.error(y_train,y_train_predicted)))
# Prediction from trained model
y_test_predicted = elm.predict(X_test)
print('Test Accuracy: ',(1-elm.error(y_test,y_test_predicted)))
#print(elm.confusion(y_test,y_test_predicted)) #value as 4E+5
y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)
'''
