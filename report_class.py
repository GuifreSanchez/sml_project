from imports import * 
from data_loader import * 
from metrics import * 
from plotter import * 
from grid_search import occ_osvm, occ_lof

class Report:
    def __init__(self, train_dm : DataManager, test_dm : DataManager, occ_method, occ_parameters, y_pred, f_pred, nu_true):
        self.train_dm = train_dm
        self.test_dm = test_dm
        self.occ_method = occ_method
        self.occ_parameters = occ_parameters
        self.y_pred = y_pred
        self.f_pred = f_pred
        self.y_true = test_dm.occ_labels
        self.report = self.get_report()
        self.nu_true = nu_true
        self.metrics = self.get_metrics()
    
    def get_report(self):
        target_names = ['Normal class', 'Outlier class']
        report = classification_report(self.y_true, self.y_pred, labels = [1, -1], target_names = target_names)
        return report 
    
    def get_metrics(self):
        y_true = self.y_true
        y_pred = self.y_pred
        f1 = custom_f1(y_true, y_pred)
        prec = custom_precision(y_true, y_pred)
        rec = custom_recall(y_true, y_pred)
        nu_prec = custom_nu_prec(y_true, y_pred, nu_true=self.nu_true)
        nu_rec = custom_nu_rec(y_true, y_pred, nu_true = self.nu_true)
        
        pr_auc = -1
        if self.occ_method == occ_osvm:
            nu = self.occ_parameters[0]
            gamma = self.occ_parameters[1]
            osvm = svm.OneClassSVM(kernel = 'rbf', gamma = gamma, nu = nu)
            osvm.fit(self.train_dm.data)
            f_pred = osvm.decision_function(self.test_dm.data)
            pr_auc = custom_pr_auc(y_true, f_pred)
        elif self.occ_method == occ_lof:
            n_neighbors = self.occ_parameters[0]
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty = True)
            lof.fit(self.train_dm.data)
            f_pred = lof.decision_function(self.test_dm.data)
            pr_auc = custom_pr_auc(y_true, f_pred)
            
            
        return np.array([f1, prec, rec, nu_prec, nu_rec, pr_auc])
    
    def get_metrics_x100(self):
        new_metrics = np.round(np.array([self.metrics[i] * 100.0 for i in range(len(self.metrics))]),2)
        return new_metrics
    
    
    
    def print_outliers(self, file_name = 'outlier_images.png', prints = False):
        # Get false negatives (in general, we are selecting number = a1, a2, etc.) and we 
        # want to see representatives of the non-a1,a2,... class
        indices = []
        n_test = len(self.test_dm.data)
        y_true = self.y_true
        y_test = self.test_dm.labels
        y_pred = self.y_pred
        f_pred = self.f_pred
        for i in range(n_test):
            if (y_true[i] == 1 and y_pred[i] == -1):
                indices.append(i)
                
        indices = np.array(indices)
        
        f_pred_fn = f_pred[indices]
        x_fn = self.test_dm.data[indices]
        #y_fn = y_true[indices]
        y_test_fn = y_test[indices]
        
        n_fn = len(x_fn)
        px_w = 16
        px_h = 16
        x_fn_images = x_fn.reshape((n_fn, px_w, px_h))
        
        n_rows = 4
        n_cols = 4
        image_size = 1.5
        # First n_rows * n_cols elements
        if prints:
            print("Number of wrongly classified n's from test set: ", n_fn)
        
        # Generate k random integers between a and b without repetition
        a = 0
        b = n_fn - 1
        k = n_rows * n_cols
        result = random.sample(range(a, b + 1), k)
        
        # Random 4x4 grid of wrongly classified 0s
        images1 = x_fn_images[result]
        f_pred1 = f_pred_fn[result]   
        y_test1 = y_test_fn[result]
        image_info1 = np.c_[f_pred1, y_test1] 
        print_images(n_rows, n_cols, image_size, images1, image_info1, file_name = "test2.png")
        
        # 5 "worst" O's
        sorted_f_indices = np.argsort(f_pred_fn)
        #print(f_pred_fn)
        #print(sorted_f_indices)
        images_fn_ord = x_fn_images[sorted_f_indices]
        f_pred_fn_ord = f_pred_fn[sorted_f_indices]
        y_test_fn_ord = y_test_fn[sorted_f_indices]
        images_info = np.c_[f_pred_fn_ord, y_test_fn_ord]
        
        n_rows = 1
        n_cols = 5
        image_size = 1.5
        
        print_images(n_rows, n_cols, image_size, images_fn_ord, images_info, file_name = file_name)