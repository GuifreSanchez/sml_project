
from imports import * 
from grid_search import * 
from data_loader import * 
from report_class import * 

def outlier_detection_usps(numbers = [0], nu = 0.05, k_fold_cv = 3, score_thr = 0.9, prints = False):
    train_dm, test_dm = get_usps_data(numbers = [0])
    nu = 0.05
    gsr = grid_search(
        train_dm, 
        score_name = nu_rec_name, 
        k_fold_cv = 3, 
        occ_method = occ_osvm,
        score_thr = 0.9,
        nu_true = nu, 
        prints = prints, 
        n_points = None)
    
    nu_gs = gsr.abv_thr_params[0]
    gamma_gs = gsr.abv_thr_params[1]
    if prints: 
        print("nu, gamma gs: ", nu_gs, gamma_gs)
    
    osvm = svm.OneClassSVM(kernel = 'rbf', gamma = gamma_gs, nu = nu_gs)
    osvm.fit(train_dm.data)
    y_pred = osvm.predict(test_dm.data)
    report = Report(train_dm, test_dm, occ_osvm, [nu_gs, gamma_gs], y_pred, nu_true = nu)
    print(report.report)
    
def outlier_detection_test():
    outlier_detection_usps(prints = True)
    
outlier_detection_test()
    
    