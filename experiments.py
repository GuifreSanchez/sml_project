
from imports import * 
from grid_search import * 
from data_loader import * 
from report_class import * 
# bc = "breast-cancer"
# shuttle = "shuttle"
# letter = "letter"
# sat = "satellite"
# pen_global = "pen-global"
# annthyroid = "annthyroid"
# aloi = "aloi"
def outlier_detection(k_fold_cv = 3, 
                    dataset = bc,
                    score_name = nu_rec_name, 
                    score_thr = 0.8, 
                    n_points = None, 
                    prints = False, 
                    gs_mode = True, 
                    occ_method = occ_osvm):
    train_dm = load_data(name = dataset)
    test_dm = load_data(name = dataset) 
    nu = train_dm.nu
    
    nu_exp = nu
    gamma_exp = 'auto'
    
    if occ_method == occ_osvm:
        if gs_mode:
            gsr = grid_search(
                train_dm, 
                score_name = score_name, 
                k_fold_cv = k_fold_cv, 
                occ_method = occ_osvm,
                score_thr = score_thr,
                nu_true = nu, 
                prints = prints, 
                n_points = n_points)
        
            nu_gs = gsr.abv_thr_params[0]
            gamma_gs = gsr.abv_thr_params[1]
            if prints: 
                print("nu, gamma gs: ", nu_gs, gamma_gs)
            
            nu_exp = nu_gs
            gamma_exp = gamma_gs
        
        
        osvm = svm.OneClassSVM(kernel = 'rbf', gamma = gamma_exp, nu = nu_exp)
        osvm.fit(train_dm.data)
        y_pred = osvm.predict(test_dm.data)
        f_pred = osvm.decision_function(test_dm.data)
        
        if prints: 
            y_pred_train = osvm.predict(train_dm.data)
            print(custom_f1(train_dm.occ_labels, y_pred_train))
            
        report = Report(train_dm, test_dm, occ_osvm, [nu_exp, gamma_exp], y_pred, f_pred, nu_true = nu)
        return report
    elif occ_method == occ_lof:
        n_neighbors_exp = 20
        if gs_mode:
            gsr = grid_search(
                train_dm, 
                score_name = score_name, 
                k_fold_cv = k_fold_cv, 
                occ_method = occ_lof,
                score_thr = score_thr,
                nu_true = nu, 
                prints = prints, 
                n_points = n_points)
        
            n_neighbors_gs = gsr.abv_thr_params[0]

            if prints: 
                print("n_neighbors = ", n_neighbors_gs)
                
            n_neighbors_exp = n_neighbors_gs
        
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors_exp, novelty=True)
        lof.fit(train_dm.data)
        y_pred = lof.predict(test_dm.data)
        f_pred = lof.decision_function(test_dm.data)
        
        if prints: 
            y_pred_train = lof.predict(train_dm.data)
            print(custom_f1(train_dm.occ_labels, y_pred_train))
            
        report = Report(train_dm, test_dm, occ_lof, [n_neighbors_exp], y_pred, f_pred, nu_true = nu)
        return report
        

def outlier_detection_usps(numbers = [0], 
                           nu = 0.05, 
                           k_fold_cv = 3, 
                           score_name = nu_rec_name, 
                           score_thr = 0.8, 
                           n_points = None, 
                           prints = False,
                           gs_mode = True, 
                           occ_method = occ_osvm):
    train_dm, test_dm = get_usps_data(numbers = numbers)
    if occ_method == occ_osvm:
        nu_exp = nu
        gamma_exp = 1.0 / (0.5 * 256) 
        if gs_mode:
            gsr = grid_search(
                train_dm, 
                score_name = score_name, 
                k_fold_cv = k_fold_cv, 
                occ_method = occ_osvm,
                score_thr = score_thr,
                nu_true = nu, 
                prints = prints, 
                n_points = n_points, 
                c_base=2.0)
        
            nu_gs = gsr.abv_thr_params[0]
            gamma_gs = gsr.abv_thr_params[1]
            
            nu_exp = nu_gs
            gamma_exp = gamma_gs
            if prints: 
                print("nu, gamma gs: ", nu_gs, gamma_gs)
        
        osvm = svm.OneClassSVM(kernel = 'rbf', gamma = gamma_exp, nu = nu_exp)
        osvm.fit(train_dm.data)
        y_pred = osvm.predict(test_dm.data)
        f_pred = osvm.decision_function(test_dm.data)
        
        if prints: 
            y_pred_train = osvm.predict(train_dm.data)
            print(custom_f1(train_dm.occ_labels, y_pred_train))
            
        report = Report(train_dm, test_dm, occ_osvm, [nu_exp, gamma_exp], y_pred, f_pred, nu_true = nu)
        return report
    
    elif occ_method == occ_lof:
        n_neighbors_exp = 20
        if gs_mode:
            gsr = grid_search(
                    train_dm, 
                    score_name = score_name, 
                    k_fold_cv = k_fold_cv, 
                    occ_method = occ_lof,
                    score_thr = score_thr,
                    nu_true = nu, 
                    prints = prints, 
                    n_points = n_points, 
                    c_base=2.0)
        
            n_neighbors_gs = gsr.abv_thr_params[0]
            if prints: 
                print("n_neighbors =  ", n_neighbors_gs)
                
            n_neighbors_exp = n_neighbors_gs
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors_exp, novelty=True)
        lof.fit(train_dm.data)
        y_pred = lof.predict(test_dm.data)
        f_pred = lof.decision_function(test_dm.data)
        
        if prints: 
            y_pred_train = lof.predict(train_dm.data)
            print(custom_f1(train_dm.occ_labels, y_pred_train))
            
        report = Report(train_dm, test_dm, occ_lof, [n_neighbors_exp], y_pred, f_pred, nu_true = nu)
        return report
    
def experiments_usps(prints = False, gs_mode = True, occ_method = occ_osvm):
    numbers_list = [
        [0],
        [2, 6],
        [0, 1, 2, 3, 4]
    ]
    
    score_thrs = [0.95, 0.95, 0.95]
    
    reports = []
    i = 0
    for numbers in numbers_list:
        score_thr = score_thrs[i]
        if prints:
            print("USPS experiment with numbers = ")
            print(numbers)
        report = outlier_detection_usps(numbers = numbers, score_name=nu_rec_name, score_thr = score_thr, gs_mode = gs_mode, occ_method = occ_method)
        reports.append(report)
        i += 1
            
    for report in reports:
        print(report.get_metrics_x100())
        
    for report in reports: 
        print(report.occ_parameters)
        
# datasets = [bc, shuttle, letter, sat, pen_global, annthyroid, aloi]
def experiments_datasets(prints = False, gs_mode = True, occ_method = occ_osvm):
    
    score_thr =0.99
    datasets = {
        bc: score_thr, 
        letter: score_thr, 
        pen_global:score_thr, 
        annthyroid:score_thr
        }
    
    reports = []
    i = 0
    for key in datasets: 
        if prints: 
            print("Outlier detection experiment dataset = ", key)
        report = outlier_detection(dataset = key, score_name = nu_rec_name, score_thr = datasets[key], gs_mode=gs_mode, occ_method = occ_method)
        reports.append(report)
        i += 1
        
    for report in reports:
        print(report.get_metrics_x100())
        
    for report in reports: 
        print(report.occ_parameters)

def outlier_detection_test():
    report = outlier_detection_usps(numbers = [0, 2, 9], prints = True)
    print(report.metrics)
    
    report.print_outliers()
    
#outlier_detection_test()

def all_experiments():
    gs_mode = True
    print("[EXPERIMENTS W/ GRID SEARCH]")
    experiments_usps(prints = True, gs_mode = gs_mode, occ_method=occ_osvm)
    experiments_datasets(prints = True, gs_mode = gs_mode, occ_method=occ_osvm)
    experiments_usps(prints = True, gs_mode = gs_mode, occ_method = occ_lof)
    experiments_datasets(prints = True, gs_mode = gs_mode, occ_method = occ_lof)
    print("\n")
    gs_mode = False
    print("[EXPERIMENTS W/O GRID SEARCH]")
    experiments_usps(prints = True, gs_mode = gs_mode, occ_method=occ_osvm)
    experiments_datasets(prints = True, gs_mode = gs_mode, occ_method=occ_osvm)
    experiments_usps(prints = True, gs_mode = gs_mode, occ_method = occ_lof)
    experiments_datasets(prints = True, gs_mode = gs_mode, occ_method = occ_lof)
    
    