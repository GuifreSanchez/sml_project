from imports import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from comparison import custom_precision, custom_recall, custom_f1, custom_score

osvm = svm.OneClassSVM()

def add_outliers(data, outliers):
    n_outliers = len(outliers)
    labels = np.concatenate([np.ones(len(data)), -np.ones(len(outliers))], axis = 0)
    num_labels = labels
    full_data = np.concatenate([data, outliers], axis=0)
    labeled_data = np.c_[full_data, labels]
    return full_data, labeled_data, labels, num_labels, n_outliers

# Here we make main data in first blob, and outliers in separated blob. 
def synthetic_data(n_samples = 150, nu = 0.1):
    n_outliers = int(n_samples * nu)
    n_inliers = n_samples - n_outliers  
    blobs_params_inliers = dict(random_state=0, n_samples=n_inliers, n_features=2)
    blobs_params_outliers = dict(random_state = 2, n_samples = n_outliers, n_features = 2)
    inliers = make_blobs(centers = [[-0.4, -0.4]], cluster_std = 0.2, **blobs_params_inliers)[0]
    outliers = make_blobs(centers = [[0.45, 0.45]], cluster_std = 0.1, **blobs_params_outliers)[0]
    return add_outliers(inliers, outliers)

# Scorers
scorer = make_scorer(custom_f1)
def new_scorer(estimator, X, y_true, **kwargs):
    y_pred = estimator.fit_predict(X)
    return custom_score(y_true, y_pred, estimator.nu)



# Grid search
k_fold_cv = 3
param_grid = {
    'nu': [(i + 1) * 0.01 for i in range(25)],
    'kernel': ['rbf'],
    'gamma': [1.0 / (2.0**(6 - i)) for i in range(7)]
}

grid_search = GridSearchCV(estimator=osvm, param_grid=param_grid, scoring=scorer, cv=k_fold_cv)

# Get data
pixel_width = 16
pixel_height = 16
X_tr, y_tr, X_te, y_te = get_usps_data()


# Set train data
n_train_r = 0.5
zeros = X_tr[y_tr == 0]
zero_labels = y_tr[y_tr == 0]
n_train = int(n_train_r * len(zeros))
train_data = zeros[:n_train]
train_labels = np.ones(n_train)


# Perform the search
grid_search.fit(train_data, train_labels)  # OCSVM only needs normal data for training
grid_search_results = grid_search.cv_results_
pd_cv_results = pd.DataFrame(grid_search_results)
columns_to_display = [
    "param_nu", "param_gamma", "mean_test_score", "rank_test_score"
]

sorted_results = pd_cv_results[columns_to_display].sort_values(by ='mean_test_score')
#print(sorted_results.head(10))
print(grid_search.best_params_)
print(grid_search.best_score_)
print(sorted_results.head(5))
print(sorted_results.tail(5))

best_gamma = grid_search.best_params_['gamma']
best_nu = grid_search.best_params_['nu']

n_test = len(X_te)
test_data = X_te[:n_test]
test_original_labels = y_te[:n_test]
test_labels = []
for i in range(n_test):
    zero = 1 if int(y_te[i]) == 0 else -1
    test_labels.append(zero) 
test_labels = np.array(test_labels)

c = 1. / best_gamma
nu = best_nu
osvm = svm.OneClassSVM(nu = nu, kernel = "rbf", gamma = 1.0 / c)
osvm.fit(zeros)
y_pred = osvm.predict(test_data)

target_names = ['Zero class', 'Non-zero class']
report = classification_report(test_labels, y_pred, labels = [1, -1], target_names = target_names)
print(report)

prec = custom_precision(test_labels, y_pred)
recall = custom_recall(test_labels, y_pred)
f1 = custom_f1(test_labels, y_pred)
print(prec, recall, f1)


test_labels = np.ones(n_train)
y_pred = osvm.predict(train_data)
target_names = ['Zero class', 'Non-zero class']
report = classification_report(test_labels, y_pred, labels = [1, -1], target_names = target_names)
print(report)

prec = custom_precision(test_labels, y_pred)
recall = custom_recall(test_labels, y_pred)
f1 = custom_f1(test_labels, y_pred)
print(prec, recall, f1)


