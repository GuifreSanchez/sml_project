from imports import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from metrics import *
from data_loader import * 


def nu_prec_scorer(estimator, X, y_true, **kwargs):
    y_pred = estimator.fit_predict(X)
    return custom_nu_prec(y_true, y_pred, estimator.nu)

occ_osvm = 'osvm'
occ_lof = 'lof'
occ_method = [occ_lof, occ_osvm]

min_nu = 0.01
max_nu = 0.05
nu_step = 0.02


c_base = 2.0
min_exp_c = -1
max_exp_c = 6

def grid_search(dm : DataManager, 
                score_name = 'f1',
                k_fold_cv = 3,
                occ_method = occ_osvm, 
                score_thr = 0.95, 
                prints = False, 
                n_points = None):
    
    if prints: 
        print("Selecting score ", score_name, "...")
    if score_name != nu_prec and score_name != pr_auc:
        scorer = make_scorer(select_score(score_name=score_name))
    elif score_name == nu_prec:
        scorer = nu_prec_scorer
    elif score_name == pr_auc:
        print("pr_auc scorer not implemented")
        return 
    
    if occ_method == occ_osvm:
        n_nus = int((max_nu - min_nu) / nu_step) + 1
        param_grid = {
            'nu': [min_nu + i * nu_step for i in range(n_nus)],
            'kernel': ['rbf'],
            'gamma': [c_base ** (-(max_exp_c - j)) for j in range(max_exp_c - min_exp_c + 1)]
        }
        osvm = svm.OneClassSVM()
        gs = GridSearchCV(estimator=osvm, param_grid=param_grid, scoring=scorer, cv=k_fold_cv)
        
        max_index = len(dm.data)
        if n_points is not None: 
            max_index = n_points
        data, occ_labels = dm.data[:max_index], dm.occ_labels[:max_index]
        
        if prints: 
            print("Performing grid search on ", n_nus * (max_exp_c - min_exp_c + 1), " parameter combinations ...")
        gs.fit(data, occ_labels)
        
        results = gs.cv_results_
        pd_results = pd.DataFrame(results)
        cols = [ "param_nu", "param_gamma", "mean_test_score", "rank_test_score"]
        
        sorted_results = pd_results[cols].sort_values(by = 'mean_test_score')
        
        best_params = np.array([gs.best_params_['nu'], gs.best_params_['gamma']])

        
        # first parameters with score value above threshold
        abv_thr_index = np.where(sorted_results['mean_test_score'] > score_thr)[0][0]
        results_thr = sorted_results.iloc[abv_thr_index]
        params_thr = np.array([results_thr['param_nu'], results_thr['param_gamma']])
        
        if prints: 
            m = 3
            print("First row abv thr = ", score_thr)
            print(results_thr)
            print("Surrounding rows")
            print(sorted_results.iloc[abv_thr_index - m:abv_thr_index + m])
            print("Best params:")
            print(best_params)
            print("First params abv thr:")
            print(params_thr)
            
        return sorted_results, best_params, params_thr, results_thr, abv_thr_index
    
    elif occ_method == occ_lof:
        print("grid search not implemented yet")
        return 
    
    
def grid_search_test():
    train_dm, test_dm = get_usps_data(numbers = [0])
    sorted_results, best_params, params_thr, results_thr, abv_thr_index = grid_search(train_dm, prints = False, score_thr=0.9)
    
# grid_search_test()
    
    
        
    
    
    