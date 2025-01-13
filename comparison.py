from imports import * 

bc = "breast-cancer"
shuttle = "shuttle"
letter = "letter"
sat = "satellite"
pen_global = "pen-global"
annthyroid = "annthyroid"
aloi = "aloi"

b1 = "blob1"
b2 = "blob2"
b3 = "blob3"

def add_outliers(data, x1, x2, n_outliers, rng: np.random.RandomState):
    outliers = rng.uniform(low=x1, high=x2, size=(n_outliers, 2))
    labels = np.concatenate([np.ones(len(data)), -np.ones(len(outliers))], axis = 0)
    #print(labels)
    num_labels = labels

    full_data = np.concatenate([data, outliers], axis=0)
    labeled_data = np.c_[full_data, labels]

    #print(labeled_data)
    #print(full_data)
    #return
    return full_data, labeled_data, labels, num_labels, n_outliers

def synthetic_data(name = b1, n_samples = 150, nu = 0.1):
    n_outliers = int(n_samples * nu)
    n_inliers = n_samples - n_outliers
    blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
    rng = np.random.RandomState(42)
    x_max = 5
    x1 = -x_max
    x2 = x_max
    if name == b1:
        inliers = make_blobs(centers = [[0, 0]], cluster_std = 0.5, **blobs_params)[0]
        return add_outliers(inliers, x1, x2, n_outliers, rng)
    elif name == b2: 
        inliers =  make_blobs(centers = [[-3, -3], [3, 3]], cluster_std = [0.1, 0.5], **blobs_params)[0]
        return add_outliers(inliers, x1, x2, n_outliers, rng)
    elif name == b3: 
        inliers = make_blobs(centers = [[-3, -3], [-3, 4], [0, 3]], cluster_std = [0.2, 0.2, 0.25], **blobs_params)[0]
        return add_outliers(inliers, x1, x2, n_outliers, rng)


# moon1 = scale_moons * (make_moons(n_samples = n_samples, noise = 0.05, random_state = 0)[0]- np.array([0.5, 0.25]))
# circles1 = scale_circles * make_circles(n_samples = n_samples, noise = 0.1, factor = 0.3, random_state = 0)[0]

datasets = [bc, shuttle, letter, sat]
synthetic = [b1, b2, b3]
def load_data(name):
    path = 'datasets/dataverse_files/' + name + '-unsupervised-ad.csv'
    # Load CSV file into a DataFrame
    df = pd.read_csv(path, header = None)
    # Exclude the last row and convert the DataFrame to a NumPy array
    data = df.iloc[:, :-1].to_numpy()
    labeled_data = df.iloc[:].to_numpy()
    labels = labeled_data[:, -1]
    num_labels = np.array([-1 if label == 'o' else +1 for label in labels])
    n_outliers = len(num_labels[num_labels == -1])
    return data, labeled_data, labels, num_labels, n_outliers

def custom_recall(y_true, y_pred):
    return custom_metric(y_true, y_pred, metric = 'recall')

def custom_precision(y_true, y_pred):
    return custom_metric(y_true, y_pred, metric = 'precision')

def custom_score(y_true, y_pred, nu):
    precision = custom_precision(y_true, y_pred)
    tp, tn, fp, fn = custom_metric(y_true, y_pred, metric = 'info')
    N = tn + fn
    P = tp + fp
    nu_pred = N / (N + P)
    nu_dist_inv = 1 / (1 + (nu - nu_pred)**2)
    return precision + nu_dist_inv
    

def custom_metric(y_true, y_pred, metric = 'precision'):
    fn = 0
    fp = 0
    tp = 0
    tn = 0
    for i in range(len(y_pred)):
        truth = y_true[i]
        pred = y_pred[i]
        if truth == -1 and pred == -1: 
            tn += 1
        if truth == -1 and pred == +1:
            fp += 1
        if truth == +1 and pred == +1: 
            tp += 1
        if truth == +1 and pred == -1: 
            fn += 1
            
            
    precision = 0.0
    recall =0.0
    f1_score = 0.0
    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = tp /(tp + fp)
        
    if (tp + fn) == 0:
        recall = 0.0
    else:
        recall = tp /(tp + fn)
    
    if (precision == 0.0) or (recall == 0.0):
        f1_score == 0.0
    else:
        f1_score = 2.0 / (1.0 / precision + 1.0 / recall)
    
    
    if metric == 'precision':
        return precision
    if metric == 'recall':
        return recall
    if metric == 'f1_score':
        return f1_score
    if metric == 'info':
        return tp, tn, fp, fn

def custom_f1(y_true, y_pred):
    return custom_metric(y_true, y_pred, metric = 'f1_score')

def custom_pr_auc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    result = auc(recall, precision)
    return result, precision, recall

def osvm_experiment(dataset_name, load_data_func, custom_nu = -1, custom_c = -1, prints = False):
    data, labeled_data, labels, num_labels, n_outliers = load_data_func(dataset_name)
    nu = 0
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
    pr_auc, precision, recall = custom_pr_auc(y_true = y_true, y_scores=y_decf)
    
    return nu, c, f1_score, pr_auc, precision, recall

def lof_experiment(dataset_name, load_data_func, n_neighbors = 2):
    data, labeled_data, labels, num_labels, n_outliers = load_data_func(dataset_name)
    
    y_true = num_labels
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    y_pred = lof.fit_predict(data)
    y_scores = lof.negative_outlier_factor_
    
    f1_score = custom_f1(y_true = y_true, y_pred = y_pred)
    pr_auc, precision, recall = custom_pr_auc(y_true = y_true, y_scores=y_scores)
    
    return n_neighbors, f1_score, pr_auc, precision, recall 


def compare_real_data(exp_real_data):
    osvm_pr_curves = {}
    lof_pr_curves = {}
    for i in range(len(exp_real_data)):
        dataset = exp_real_data[i]
        nu, c, osvm_f1_score, osvm_pr_auc, osvm_precision, osvm_recall = osvm_experiment(dataset, load_data, prints = False)
        n_neighbors, lof_f1_score, lof_pr_auc, lof_precision, lof_recall = lof_experiment(dataset, load_data)
        table_name = dataset
        if dataset == bc:
            table_name = "bc"
        if dataset == annthyroid:
            table_name = "antrd"
        if dataset == pen_global: 
            table_name = "peng"
        print(table_name + "\t", "%.2f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f" % (nu * 100, osvm_f1_score, osvm_pr_auc, n_neighbors, lof_f1_score, lof_pr_auc))
        osvm_pr_curves[dataset] = np.c_[osvm_recall, osvm_precision]
        lof_pr_curves[dataset] = np.c_[lof_recall, lof_precision]
    return osvm_pr_curves, lof_pr_curves

def compare_synth_data(exp_synth_data):
    exp_data = exp_synth_data
    osvm_pr_curves = {}
    lof_pr_curves = {}
    for i in range(len(exp_data)):
        dataset = exp_data[i]
        nu, c, osvm_f1_score, osvm_pr_auc, osvm_precision, osvm_recall = osvm_experiment(dataset, synthetic_data, prints = False)
        n_neighbors, lof_f1_score, lof_pr_auc, lof_precision, lof_recall = lof_experiment(dataset, synthetic_data,)
        table_name = dataset
 
        print(table_name + "\t", "%.2f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f" % (nu * 100, osvm_f1_score, osvm_pr_auc, n_neighbors, lof_f1_score, lof_pr_auc))
        osvm_pr_curves[dataset] = np.c_[osvm_recall, osvm_precision]
        lof_pr_curves[dataset] = np.c_[lof_recall, lof_precision]
        
    return osvm_pr_curves, lof_pr_curves
def header():
    print("-------\t OSVM\t \t\t LOF")
    print("dataset\t nu\t f1\t auc\t k_nbhs\t f1\t auc")


def dim_color(hex_color, factor=0.8):
    """
    Returns a slightly dimmer version of a given hex color.

    Parameters:
        hex_color (str): The hex color string (e.g., "#17becf").
        factor (float): The dimming factor (less than 1 to dim, 1 retains original brightness).

    Returns:
        str: The dimmed hex color string.
    """
    # Remove the '#' and convert to integers
    hex_color = hex_color.lstrip('#')
    rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

    # Apply the dimming factor and ensure values stay in range
    dimmed_rgb = [max(0, min(255, int(c * factor))) for c in rgb]

    # Convert back to hex
    return '#' + ''.join(f'{c:02x}' for c in dimmed_rgb)

# Example usage
original_color = "#17becf"
dimmer_color = dim_color(original_color)
print(f"Original: {original_color}, Dimmer: {dimmer_color}")


def compute_results():
    exp_real_data = [bc, letter,annthyroid, pen_global]
    exp_synth_data = [b1, b2 ,b3]
    header()
    
    osvm_pr_curves_real, lof_pr_curves_real = compare_real_data(exp_real_data=exp_real_data)
    osvm_pr_curves_synth, lof_pr_curves_synth = compare_synth_data(exp_synth_data=exp_synth_data)
    
    osvm_pr_curves = osvm_pr_curves_real.copy()
    osvm_pr_curves.update(osvm_pr_curves_synth)
    lof_pr_curves = lof_pr_curves_real.copy()
    lof_pr_curves.update(lof_pr_curves_synth)
    
    plt.figure(figsize=(10, 6))

    # Iterate through dictionary and plot each precision-recall curve
    
    colors = {
        bc: "#1f77b4",        # Blue
        letter:"#d62728",    # Red
        annthyroid: "#ff7f0e",# Orange
        pen_global: "#8c564b",# Brown
        b1: "#9467bd",        # Purple
        b2: "#2ca02c",        # Green
        b3: "#17becf"         # Cyan
    }

    for key, value in osvm_pr_curves.items():
        if key is b2 or key is bc or key is annthyroid: 
            osvm_curve = osvm_pr_curves[key]
            lof_curve = lof_pr_curves[key]
            osvm_recall = osvm_curve[:, 0]     # First column: recall
            osvm_precision = osvm_curve[:, 1]  # Second column: precision
            lof_recall = lof_curve[:, 0]     # First column: recall
            lof_precision = lof_curve[:, 1]  # Second column: precision
            color = colors[key]
            label = "synthetic" if (key is b1 or key is b2 or key is b3) else key
            plt.plot(osvm_recall, osvm_precision, linestyle = '-', label=label + " (osvm)", color = color)
            plt.plot(lof_recall, lof_precision, linestyle = '--', label=label + " (lof)", color = dim_color(color))

    # Customize the plot
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    # plt.ylim([0.8, 1.0])
    # plt.xlim([0.0, 1.0])
    plt.legend()  # Add a legend
    plt.grid(True)
    plt.savefig('pr_curves_general.png', dpi = 300)
    # Show the plot
    plt.show()
   
    
    #compare_synth_data(exp_synth_data=exp_synth_data)
    
#compute_results()






