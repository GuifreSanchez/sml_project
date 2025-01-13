from imports import * 
from comparison import custom_f1, custom_pr_auc
from osvm_2d import plot_2d

b1 = "blob1"
b2 = "blob2"
b3 = "blob3"

def add_outliers(data, outliers):
    n_outliers = len(outliers)
    labels = np.concatenate([np.ones(len(data)), -np.ones(len(outliers))], axis = 0)
    num_labels = labels
    full_data = np.concatenate([data, outliers], axis=0)
    labeled_data = np.c_[full_data, labels]

    return full_data, labeled_data, labels, num_labels, n_outliers


def synthetic_data(name = b1, n_samples = 150, nu = 0.1):
    n_outliers = int(n_samples * nu)
    n_inliers = n_samples - n_outliers  
    blobs_params_inliers = dict(random_state=0, n_samples=n_inliers, n_features=2)
    blobs_params_outliers = dict(random_state = 2, n_samples = n_outliers, n_features = 2)
    inliers = make_blobs(centers = [[-0.4, -0.4]], cluster_std = 0.2, **blobs_params_inliers)[0]
    outliers = make_blobs(centers = [[0.45, 0.45]], cluster_std = 0.1, **blobs_params_outliers)[0]
    return add_outliers(inliers, outliers)
    

def osvm_experiment(dataset_name, load_data_func, n_samples = 150, nu_synthetic = 0.1, custom_nu = -1, custom_c = -1, prints = False):
    
    data, labeled_data, labels, num_labels, n_outliers = load_data_func(dataset_name, n_samples = n_samples, nu = nu_synthetic)
    nu = 0
    n_inliers = len(data) - n_outliers
    if custom_nu > 0: 
        nu = custom_nu
    else:
        nu = n_outliers / float(len(data))
    
    if prints: 
        print("n_outliers, len data, nu", n_outliers, len(data), nu)
    c = -1
    if custom_c > 0: 
        c = custom_c
        
    y_true = num_labels
    if custom_c > 0: 
        osvm = svm.OneClassSVM(nu=nu, kernel="rbf", gamma = 1.0 / float(custom_c))
    else:
        osvm = svm.OneClassSVM(nu = nu, kernel = 'rbf', gamma = 'auto')
    
    y_pred = osvm.fit_predict(data)
    y_decf = osvm.decision_function(data)
    y_scores = osvm.score_samples(data)
    #target_names = ["Normal class", "Outlier class"]
    
    f1_score = custom_f1(y_true = y_true, y_pred = y_pred)
    
    pr_auc = custom_pr_auc(y_true = y_true, y_scores=y_decf)
    x_range = [-7, 7]
    delta_t = 0
    return osvm, data, y_pred, x_range, n_inliers, n_outliers, delta_t, nu, c, f1_score, pr_auc








nus_osvm = [0.2]
cs_osvm = [0.01, 0.25]

from itertools import product

n_samples = 150
nu_data = 0.1

n_rows = len(nus_osvm)
n_cols= len(cs_osvm)
image_size = 5
#fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(int(image_size * n_rows), int(image_size * n_cols)))

# i = 0
# generate = True
# if generate: 
#     for nu_osvm, c_osvm in product(nus_osvm, cs_osvm):
#         file_name = "blob_2_nu_" + str(nu_osvm) + "_" + "c_" + str(c_osvm)
#         osvm, data, y_pred, x_range, n_inliers, n_outliers, delta_t, nu_osvm, c_osvm, f1_score, pr_auc = osvm_experiment(b1, synthetic_data, custom_nu = nu_osvm, custom_c = c_osvm, prints = False)
#         path = file_name + ".png"
#         print("Generating plot ", path)
#         x_range = [-1, 1]
#         plot_2d(osvm, data, y_pred, x_range, n_inliers, n_outliers, delta_t, 
#                         path = path, 
#                         grid_size = 200, 
#                         image_size = image_size, 
#                         print_levels = False, 
#                         print_density = False, 
#                         print_points = True, nu = nu_osvm, c = c_osvm)
    
# import matplotlib.image as mpimg

# # Load the image
# # Create a 4x4 grid
# image_size = 1

# height = 4
# width = height * 3
# ncols = 2
# nrows = 2
# fig, axes = plt.subplots(nrows, ncols, figsize = (10, 10))
# adjust = 2.01
# plt.subplots_adjust(wspace=-adjust, hspace=-adjust) 
# i = 0
# for nu_osvm in nus_osvm: 
#     j = 0
#     for c_osvm in cs_osvm:
#         img = mpimg.imread('blob_2_nu_'+str(nu_osvm) + '_c_'+str(c_osvm)+'.png')
#         axes[i, j].imshow(img)
#         axes[i, j].axis('off')
#         j += 1
#     i += 1
    


# # Adjust layout
# plt.tight_layout()
# plt.savefig('slides.png', dpi = 300)
# plt.show()