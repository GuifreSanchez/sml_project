import h5py
import time

import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import SGDOneClassSVM 

# Customize font settings (e.g., using 'serif' font)
plt.rcParams.update({
    'font.family': 'sans-serif',     # Font family (serif, sans-serif, etc.)
    'font.size': 10,            # Font size
    'font.weight': 'normal',      # Font weight (normal, bold, etc.)
    'axes.titlesize': 12,       # Title font size
    'axes.labelsize': 10,       # Axis labels font size
    'xtick.labelsize': 6,      # X-axis tick label size
    'ytick.labelsize': 6       # Y-axis tick label size
})


def get_usps_data():
    path = 'datasets/usps.h5'
    with h5py.File(path, 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
    return X_tr, y_tr, X_te, y_te
        

X_tr, y_tr, X_te, y_te = get_usps_data()

pixel_width = 16
pixel_height = 16
n_samples = len(X_tr)
print(X_tr.shape)

X_tr_images = X_tr.reshape((n_samples, pixel_width, pixel_height))
image_size = 1.25
n_rows = 1
n_cols = 5

def print_images(n_rows, n_cols, image_size, images, labels, file_name = 'plot.png'):
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(int(0.6 * image_size * n_cols), int(image_size * n_rows)))
    plt.subplots_adjust(wspace = 0.2)
    for ax, image, label in zip(axes.ravel(), images[:n_rows * n_cols], labels[:n_rows * n_cols]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(" %.2f (%i) " % (label[0], label[1]))
    
    plt.tight_layout()
    plt.savefig(file_name, dpi = 300)
    plt.show()
    
    plt.close(fig)
    
        
#print_images(n_rows, n_cols, image_size, X_tr_images, y_tr)
    
zeros = X_tr[y_tr == 0]
n_zeros = len(zeros)
zero_images = zeros.reshape((n_zeros, pixel_width, pixel_height))
zero_labels = y_tr[y_tr == 0]
#print_images(n_rows, n_cols, image_size, zero_images, zero_labels)
#print(zeros.shape)
#print(X_te.shape)




def experiment_test_set(full_test_data, full_test_labels, fraction_test = 1.0, scale = False, c = 0.5 * 256, nu = 0.05, use_labels = False):
    total_test = len(full_test_data)
    n_test = int(fraction_test * total_test)
    x_test_no_label = full_test_data[:n_test]
    y_test = full_test_labels[:n_test]
    if use_labels:
        x_test = []
        for i in range(n_test):
            label_vector = np.zeros(10)
            label_vector[y_test[i]] = 1.0
            x_test.append(np.concatenate([x_test_no_label[i], label_vector]))
        x_test = np.array(x_test)
    else:
        x_test = x_test_no_label

    # print(x_test[0].shape, y_test[0])
    # print(x_test[0][250:])
    # return 0
    
    
    if scale: 
        scaler = StandardScaler()
        x_test = scaler.fit_transform(x_test)
        
        
    print(x_test.shape, len(x_test))
    osvm = svm.OneClassSVM(nu = nu, kernel = "rbf")
    osvm.fit(x_test)
    y_pred = osvm.predict(x_test)
    f_pred = osvm.decision_function(x_test)
    
    image_info = []
    
    true_positives = 0
    for i in range(n_test):
        image_info.append([int(y_test[i]), y_pred[i], f_pred[i], +1])
        # Compute fraction of numbers correctly identified with osvm (i.e. in the positive class)
        if (y_pred[i] == 1):
            true_positives += 1
    
    fraction_true_positives =  true_positives / n_test
    image_info = np.array(image_info)
    print("# n's in test set: ", n_test)
    print("# n's from test set correctly identified by OSVM: ", true_positives)
    print("Fraction of 'True positives': ", fraction_true_positives)
    
    y_true = image_info[:,3]
    target_names = ["Normal class", "Outlier class"]
    #print(y_true)
    #print(y_pred)
    # The problem in this case is that we do not have labeled negative instances! 
    report = classification_report(y_true, y_pred, labels = [1, -1], target_names = target_names)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, -1], normalize= "true")
    
    print(report)
    print(conf_matrix)
    
    return x_test_no_label, y_pred, f_pred, image_info, y_true, report, conf_matrix
    
    


def experiment_zeros(full_zeros_train,
                     full_test_data, 
                     full_test_labels, 
                     fraction_zeros = 1.0, 
                     fraction_test = 1.0, 
                     scale = False,
                     c = 0.5 * 256, 
                     nu = 0.05):
    # number of train set 0s: 1194
    # test set size: 2007

    total_zeros = len(full_zeros_train)
    total_test = len(full_test_data)
    n_zeros = int(fraction_zeros * total_zeros)
    n_test = int(fraction_test * total_test)
    x_train = full_zeros_train[:n_zeros]
    print(x_train.shape)
    x_test = full_test_data[:n_test]
    y_test = full_test_labels[:n_test]
    
    if scale: 
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
    osvm = svm.OneClassSVM(nu = nu, kernel = "rbf", gamma = 1.0 / c)
    osvm.fit(x_train)
    y_pred = osvm.predict(x_test)
    f_pred = osvm.decision_function(x_test)
    
    #print("Decision function value on test data: ", f_pred[0])
    #print(y_pred)

    #test_images = x_test.reshape((n_test, 28, 28))
    
    # osvm class predicition, decision function value, label from test data,
    # anomaly detection result based on true labels
    image_info = []
    
    n_test_zeros = 0
    true_zeros = 0
    for i in range(n_test):
        occ_from_labels = +1 if int(y_test[i]) == 0 else -1
        image_info.append([int(y_test[i]), y_pred[i], f_pred[i], occ_from_labels])
        # Compute fraction of 0's correctly identified with osvm
        if int(y_test[i]) == 0:
            n_test_zeros += 1
            if (y_pred[i] == 1):
                true_zeros += 1
    
    fraction_true_zeros =  true_zeros / n_test_zeros
    image_info = np.array(image_info)
    print("# 0's in test set: ", n_test_zeros)
    print("# 0's from test set correctly identified by OSVM: ", true_zeros)
    print("Fraction of 'True positives': ", fraction_true_zeros)
    
    y_true = image_info[:,3]
    target_names = ["Normal class", "Outlier class"]
    #print(y_true)
    #print(y_pred)
    report = classification_report(y_true, y_pred, labels = [1, -1], target_names = target_names)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, -1], normalize= "true")
    
    print(report)
    print(conf_matrix)
    
    return x_test, y_pred, f_pred, image_info, y_true, report, conf_matrix
    
def print_outliers(results, file_name = 'plot.png'):
    x_test, y_pred, f_pred, image_info, y_true, report, conf_matrix = results
    #print(y_pred)
    #print(y_true)
    #print(y_pred.shape, y_true.shape)
    # get false negative indices
    indices = []
    n_test = len(x_test)
    for i in range(n_test):
        if (y_true[i] == 1 and y_pred[i] == -1):
            indices.append(i)
    # Gather identified as outliers indices
    indices = np.array(indices)
    
    f_pred_fn = f_pred[indices]
    x_fn = x_test[indices]
    y_fn = y_true[indices]
    image_info_fn = image_info[indices]
    
    n_fn = len(x_fn)
    px_w = 16
    px_h = 16
    x_fn_images = x_fn.reshape((n_fn, px_w, px_h))
    
    n_rows = 2
    n_cols = 2
    image_size = 1.5
    # First n_rows * n_cols elements
    
    print("Number of wrongly classified n's from test set: ", n_fn)
    
    # Generate k random integers between a and b without repetition
    a = 0
    b = n_fn - 1
    k = n_rows * n_cols
    result = random.sample(range(a, b + 1), k)
    
    # Random 4x4 grid of wrongly classified 0s
    images = x_fn_images[result]
    labels = f_pred_fn[result]    
    #print_images(n_rows, n_cols, image_size, images, labels)
    
    # 5 "worst" O's
    sorted_f_indices = np.argsort(f_pred_fn)
    #print(f_pred_fn)
    #print(sorted_f_indices)
    images_ord = x_fn_images[sorted_f_indices]
    labels_ord = f_pred_fn[sorted_f_indices]
    image_info_fn_ord = image_info_fn[sorted_f_indices]
    n_rows = 1
    n_cols = 5
    image_size = 1.5
    
    print(image_info_fn_ord[:n_cols])
    #print_images(n_rows, n_cols, image_size, images_ord, labels_ord, file_name = file_name)
    
    n_rows = 4
    n_cols = 6
    a = 0
    b = len(x_test)
    k = n_rows * n_cols
    result = random.sample(range(a, b + 1), k)
    images2 = x_test[result].reshape((k, px_w, px_h))
    labels2 = np.c_[f_pred[result], image_info[result, 0]]
    print_images(n_rows, n_cols, image_size,images2,labels2, file_name = "random_sample.png")
    
    
usps_zeros_results = experiment_zeros(full_zeros_train=zeros,full_test_data=X_te, full_test_labels=y_te)
#usps_test_results = experiment_test_set(full_test_data = X_te, full_test_labels = y_te, use_labels=True, scale = False)
#print_outliers(usps_zeros_results, file_name = 'outlier_zeros.png')
#print_outliers(usps_test_results, file_name = 'outlier_hw_digits.png')


