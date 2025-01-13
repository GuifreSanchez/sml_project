from sklearn.datasets import make_blobs, make_moons, make_circles
import time
#from comparison import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.datasets import make_blobs, make_moons, make_circles

from sklearn.linear_model import SGDOneClassSVM 

# Customize font settings (e.g., using 'serif' font)
plt.rcParams.update({
    'font.family': 'sans-serif',     # Font family (serif, sans-serif, etc.)
    'font.size': 12,            # Font size
    'font.weight': 'normal',      # Font weight (normal, bold, etc.)
    'axes.titlesize': 14,       # Title font size
    'axes.labelsize': 10,       # Axis labels font size
    'xtick.labelsize': 8,      # X-axis tick label size
    'ytick.labelsize': 8       # Y-axis tick label size
})

def random_s1(n_points=100, theta1 =0, theta2 = 1.0):
    # Generate n_points random angles uniformly distributed between 0 and 2π
    angles = np.random.uniform(theta1 * 2 * np.pi, theta2 * 2 * np.pi, n_points)
    
    # Compute the x and y coordinates of the points on the unit circle
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Return the points as a 2D array
    return np.column_stack((x, y))



def experiment_2d(data, n_samples = 400, nu = 0.15, c = 10.0, x_range = [-7, 7]):
    n_outliers = int(nu * n_samples)
    n_inliers = n_samples - n_outliers
    
    blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
    blob2 = make_blobs(centers = [[-3, -3], [3, 3]], cluster_std = [1.0, 1.0], **blobs_params)[0]
    outliers = 7.0 * random_s1(n_outliers, theta1 = 0.3, theta2 = 0.45)
    s1_noise = 0.5
    for i in range(len(outliers)):
        n1 = np.random.normal(loc = 0, scale = s1_noise, size = 1)
        n2 = np.random.normal(loc = 0, scale = s1_noise, size = 1)
        print(n1)
        outliers[i][0] += n1[0]
        outliers[i][1] += n2[0]
    data = blob2
    
    osvm = svm.OneClassSVM(nu = nu, kernel = "rbf", gamma = 1.0 / c)
    
    X = data
    rng = np.random.RandomState(42)
    x1 = x_range[0] + 1
    x2 = x_range[1] - 1
    #X = np.concatenate([data, rng.uniform(low=x1, high=x2, size=(n_outliers, 2))], axis=0)
    X = np.concatenate([data, outliers], axis=0)
    t0 = time.time()
    osvm.fit(X)
    t1 = time.time()
    y_pred = osvm.predict(X)
    delta_t = t1 - t0
    
    return osvm, X, y_pred, x_range, n_inliers, n_outliers, delta_t

def plot_2d(osvm: svm.OneClassSVM,  X, y_pred, x_range, n_inliers, n_outliers, delta_t, path = "standard_plot.png", grid_size = 200, image_size = 5,
            print_levels = False, 
            print_density = False,
            print_points = True, nu = 0.1, c = 10.0):
    x1 = x_range[0]
    x2 = x_range[1]
    x = np.linspace(x1, x2, grid_size)
    y = np.linspace(x1, x2, grid_size)
    xx, yy = np.meshgrid(x,y)
    n_rows = 1
    n_cols = 1
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(int(image_size * n_rows), int(image_size * n_cols)))

    W = osvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    W = W.reshape(xx.shape)

    if print_levels: 
        outlier_point = np.array([x1, x1])
        outlier_f = np.abs(osvm.decision_function(np.array([outlier_point]))[0])
        # print(outlier_f)
        n_levels = 15
        levels = [-outlier_f + 2 * outlier_f * float(i) / n_levels for i in range(n_levels + 1)]
        axes.contour(xx, yy, W, levels = levels, linewidth = 2, cmap = 'inferno')
        
    if print_density:
        axes.imshow(W, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='Greys')
        
    if print_points:
        #print(n_inliers)
        lw1 = 0.75
        lw2 = 1.0
        cw = 1.25
        color = 'White' if print_density else 'Black'
        axes.scatter(X[:n_inliers,0],X[:n_inliers,1], s = 100, linewidth = lw2,marker = '+', color= "green")
        axes.scatter(X[n_inliers:,0],X[n_inliers:,1], s = 100, linewidth = lw2,marker = '+', color= "red")
        colors = np.array(["red", "green"])
        axes.scatter(X[:, 0], X[:,1], s = 50, linewidth = lw1, color = colors[(y_pred + 1) // 2], marker = 'o', facecolors = 'none') # color dependent on prediction
        # Print 0 dec. func. level always
        axes.contour(xx, yy, W, levels = [0], linewidths = [cw], linestyles = ['solid'], colors = color)
    #axes.set_title("nu = %.2f, c = %.2f" % (nu, c))
    axes.set_title(r"$\nu = " + f"{nu:.2f}" + r", c = " + f"{c:.2f}" + r"$")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_xlim(x1, x2)
    axes.set_ylim(x1, x2)
    # axes.text(0.99,
    #         0.01,
    #             ("%.4f s" % (delta_t)).lstrip("0"),
    #             transform=plt.gca().transAxes,
    #             size=15,
    #             horizontalalignment="right",)

    # Save only the plot associated with the axes
    #extent = axes.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, dpi = 300)
    
    
    plt.show()
    return fig, axes
    #plt.close(fig)



n_samples= 100
nu = 0.15
c = 10.0
n_outliers = int(nu * n_samples)
n_inliers = n_samples - n_outliers


scale_moons = 4.0
scale_circles = 6.0
# data sets
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
blob1 = make_blobs(centers = [[0, 0]], cluster_std = 0.5, **blobs_params)[0]
blob2 = make_blobs(centers = [[-2, -2], [3, 3]], cluster_std = [0.1, 0.75], **blobs_params)[0]
blob3 = make_blobs(centers = [[2, 2], [-2, 2], [0, 0]], cluster_std = [1.0, 0.8, 0.5], **blobs_params)[0]
moon1 = scale_moons * (make_moons(n_samples = n_samples, noise = 0.05, random_state = 0)[0]- np.array([0.5, 0.25]))
circles1 = scale_circles * make_circles(n_samples = n_samples, noise = 0.1, factor = 0.3, random_state = 0)[0]
uniform1 = 14.0 * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)
datasets = [
    #make_blobs(centers = [[0, 0], [1, 1]], cluster_std=0.5, **blobs_params)[0]
    blob1,
    blob2, 
    moon1, 
    circles1, 
    uniform1
]

datasets_names = ["blob1", "blob2", "moon1", "circles1", "uniform1"]


data = datasets[1] # datasets depend on nu!! 

nus = [0.01, 0.1, 0.5]
cs = [0.5, 1.0, 10.0]

from itertools import product


n_samples = 500

# n_rows = len(nus)
# n_cols= len(cs)
# image_size = 5
# fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(int(image_size * n_rows), int(image_size * n_cols)))
# axes = axes.flatten()
# i = 0
# x1 = -7.5
# x2 = 7.5
# for nu, c in product(nus, cs):
#     file_name = "blob_2_nu_" + str(nu) + "_" + "c_" + str(c)
#     osvm, X, y_pred, x_range, n_inliers, n_outliers, delta_t = experiment_2d(data, n_samples = n_samples, nu = nu, c = c, x_range = [x1, x2])
#     path = file_name + ".png"
#     print("Generating plot ", path)
#     _, ax = plot_2d(osvm, X, y_pred, x_range, n_inliers, n_outliers, delta_t, path = path, grid_size = 200, image_size = 5,
#                print_levels = False, 
#                print_density = False,
#                print_points = True)

    
    


