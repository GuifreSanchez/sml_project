from imports import * 
from data_loader import * 


def simple_plot2d(data2d : DataManager, 
                  grid_size = 200,
                  image_size = 5,
                  plot_name = "test_plot", 
                  show = True):
    if len(data2d.data[0]) != 2:
        print("not 2d data")
        return 
    
    data = data2d.data
    nu = data2d.nu
    xs = data[:,0]
    ys = data[:,1]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    
    # x = np.linspace(x_min, x_max, grid_size)
    # y = np.linspace(y_min, y_max, grid_size)
    # xx, yy = np.meshgrid(x,y)
    n_rows = 1
    n_cols = 1
    
    inliers = data[np.where(data2d.occ_labels == +1)]
    outliers = data[np.where(data2d.occ_labels == -1)]
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(int(image_size * n_rows), int(image_size * n_cols)))
    lw1 = 0.75
    lw2 = 1.0
    cw = 1.25
    axes.scatter(inliers[:,0],inliers[:,1], s = 100, linewidth = lw2,marker = '+', color= "green")
    axes.scatter(outliers[:,0],outliers[:,1], s = 100, linewidth = lw2,marker = '+', color= "red")
    colors = np.array(["red", "green"])

    axes.set_title(r"$\nu = " + f"{nu:.2f}" + r"$")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)


    path = "simple_plot2d/" + plot_name + ".png"
    fig.savefig(path, dpi = 300)
    
    if show:
        plt.show()
    
    
    
def print_images(n_rows, n_cols, image_size, images, image_info, file_name = 'plot.png'):
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(int(0.6 * image_size * n_cols), int(image_size * n_rows)))
    plt.subplots_adjust(wspace = 0.2)
    for ax, image, label in zip(axes.ravel(), images[:n_rows * n_cols], image_info[:n_rows * n_cols]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        a = label[0]
        b = label[1]
        ax.set_title(" %.2f (%i) " % (a, b))
    
    plt.tight_layout()
    plt.savefig(file_name, dpi = 300)
    plt.show()
    plt.close(fig)
    
    
def simple_plot2d_test():
    synth1 = synth_2_modes()
    synth2 = synth_moons()
    simple_plot2d(synth2)
    
#simple_plot2d_test()
    

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