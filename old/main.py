import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.datasets import make_blobs, make_moons, make_circles

from sklearn.linear_model import SGDOneClassSVM 

# Customize font settings (e.g., using 'serif' font)
plt.rcParams.update({
    'font.family': 'sans-serif',     # Font family (serif, sans-serif, etc.)
    'font.size': 16,            # Font size
    'font.weight': 'normal',      # Font weight (normal, bold, etc.)
    'axes.titlesize': 16,       # Title font size
    'axes.labelsize': 14,       # Axis labels font size
    'xtick.labelsize': 12,      # X-axis tick label size
    'ytick.labelsize': 12       # Y-axis tick label size
})



n_samples = 400
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers


anomaly_algorithms = [
    (
        "One-Class SVM", svm.OneClassSVM(nu = outliers_fraction, kernel = "rbf", gamma = 0.1)
        
    )
]

scale_moons = 4.0
scale_circles = 6.0
# data sets
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
blob1 = make_blobs(centers = [[0, 0]], cluster_std = 0.5, **blobs_params)[0]
blob2 = make_blobs(centers = [[2, 2], [-2, 2]], cluster_std = [0.5, 0.5], **blobs_params)[0]
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

# grid for the predictions and drawing the contour lines
# xx/yy -> 2d array where each row/column is a copy of the x/y-coordinates from linspace array
# xx, yy give all coordinates of points in resulting grid, following: 
# print(xx[i, j], yy[i, j])
# print(a+ (b - a)/(n-1) * j, a+ (b - a)/(n-1) * i)

grid_size = 200

xx, yy = np.meshgrid(np.linspace(-7, 7, grid_size), np.linspace(-7, 7, grid_size))


fig_height = 50
fig_width = fig_height / len(datasets)
plt.figure(figsize = (len(anomaly_algorithms) * fig_width, fig_height))

plt.subplots_adjust(
    left = 0.02, right = 0.98, bottom = 0.001, top = 0.96, wspace = 0.05, hspace = 0.01
)

plot_num = 1
rng = np.random.RandomState(42)

for i_dataset, X in enumerate(datasets):
    # add outliers, uniform random noise
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)

    i_algorithm = 0
    for name, algorithm in anomaly_algorithms: 
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        
        #plot = plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
        plt.figure(figsize = (fig_width, fig_width))
        if i_dataset == 0:
            plt.title(name, size = 18)
            
        # fit data, tag outliers
        y_pred = algorithm.fit(X).predict(X)
        
        # plot level lines and points
        Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) # stack the raveled 1d arrays column-wise
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels = [0], linewidth = 2, colors = "black")
        
        colors = np.array(["red", "green"])
        plt.scatter(X[:, 0], X[:,1], s = 50, color = colors[(y_pred + 1) // 2], marker = 'o', facecolors = 'none') # color dependent on prediction

        plt.scatter(X[:n_inliers,0],X[:n_inliers,1], s = 100, marker = '+', color= "green")
        plt.scatter(X[n_inliers:,0],X[n_inliers:,1], s = 100, marker = '+', color= "red")
        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        plot_num += 1
        i_algorithm += 1
        plt.savefig(datasets_names[plot_num -2] + ".png", dpi=300, bbox_inches='tight')
        plt.close()
    

plt.savefig("osvm_2d_examples.png", dpi=300, bbox_inches='tight')

print("Subplots saved as separate images.")

#plt.show()
