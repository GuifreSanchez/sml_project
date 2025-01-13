from imports import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from metrics import *
from data_loader import * 



occ_osvm = 'osvm'
occ_lof = 'lof'
occ_method = [occ_lof, occ_osvm]

class GridSearchResults:
    def __init__(self, sorted_results, best_params, score_name, score_thr, occ_method):
        self.sorted_results = sorted_results
        self.best_params = best_params
        self.score_name = score_name
        self.score_thr = score_thr
        self.occ_method = occ_method
        self.abv_thr_success = False
        self.abv_thr_index = self.get_abv_thr_index(prints = False)
        self.abv_thr_params = self.get_abv_thr_params()
        
    
    def get_abv_thr_index(self,prints = False):
        if prints: 
            print(self.sorted_results.tail(10))
        abv_thr_indices = np.where(self.sorted_results['mean_test_score'] > self.score_thr)[0]
        if len(abv_thr_indices) == 0:
            if prints: 
                print("No score value above thr = ", self.score_thr)
        else:
            self.abv_thr_success = True
            
        abv_thr_index = len(self.sorted_results) - 1
        if self.abv_thr_success == True:
            abv_thr_index = np.array(np.where(self.sorted_results['mean_test_score'] > self.score_thr)[0])[0]
            
        return abv_thr_index
    
    def get_abv_thr_params(self, prints = False):
        if self.occ_method == occ_osvm:
            row = self.sorted_results.iloc[self.abv_thr_index]
            abv_thr_params = np.array([row['param_nu'], row['param_gamma']])
            if prints: 
                m = 3
                print("First row abv thr = ", self.score_thr)
                print(self.sorted_results.iloc[self.abv_thr_index])
                print("Surrounding rows")
                print(self.sorted_results.iloc[self.abv_thr_index - m:self.abv_thr_index + m])
                print("Best params:")
                print(self.best_params)
                print("First params abv thr:")
                print(abv_thr_params)
            return abv_thr_params
        elif self.occ_method == occ_lof: 
            row = self.sorted_results.iloc[self.abv_thr_index]
            abv_thr_params = np.array([row['param_n_neighbors']])
            return abv_thr_params
    
    
MAX_POINTS = 1500

def grid_search(dm : DataManager, 
                score_name = 'f1',
                k_fold_cv = 3,
                occ_method = occ_osvm, 
                score_thr = 0.95,
                nu_true = 0.05,  
                prints = False, 
                n_points = None, 
                min_nu = 0.02,
                max_nu = 0.2, 
                nu_step = 0.02, 
                c_base = 10.0,
                min_exp_c = 0,
                max_exp_c = 5):
    
    if prints: 
        print("Selecting score ", score_name, "...")
    if score_name != pr_auc_name:
        scorer = make_scorer(select_score(score_name=score_name, nu_true = nu_true))
    elif score_name == pr_auc_name:
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
        
        max_index = np.min([len(dm.data),MAX_POINTS])
        if n_points is not None: 
            max_index = np.min([n_points, MAX_POINTS])
        data, occ_labels = dm.data[:max_index], dm.occ_labels[:max_index]
        
        if prints: 
            print("Performing grid search on ", n_nus * (max_exp_c - min_exp_c + 1), " parameter combinations ...")
        gs.fit(data, occ_labels)
        
        results = gs.cv_results_
        pd_results = pd.DataFrame(results)
        cols = [ "param_nu", "param_gamma", "mean_test_score", "rank_test_score"]
        
        sorted_results = pd_results[cols].sort_values(by = 'mean_test_score')
        
        best_params = np.array([gs.best_params_['nu'], gs.best_params_['gamma']])
        
        final_results = GridSearchResults(sorted_results, best_params, score_name=score_name, score_thr=score_thr, occ_method=occ_method)
        return final_results
    
    elif occ_method == occ_lof:
        n_neighbors_params = [5, 10, 20, 50, 100, 200]
        param_grid = {
            'n_neighbors': n_neighbors_params,
        }
        lof = LocalOutlierFactor(novelty=True)
        gs = GridSearchCV(estimator=lof, param_grid=param_grid, scoring=scorer, cv=k_fold_cv)
        
        max_index = np.min([len(dm.data),MAX_POINTS])
        if n_points is not None: 
            max_index = np.min([n_points, MAX_POINTS])
        data, occ_labels = dm.data[:max_index], dm.occ_labels[:max_index]
        
        if prints: 
            print("Performing grid search on ", len(n_neighbors_params), " parameter combinations ...")
        
        gs.fit(data, occ_labels)
        
        results = gs.cv_results_
        pd_results = pd.DataFrame(results)
        cols = [ "param_n_neighbors", "mean_test_score", "rank_test_score"]
        
        sorted_results = pd_results[cols].sort_values(by = 'mean_test_score')
        
        best_params = np.array([gs.best_params_['n_neighbors']])
        
        final_results = GridSearchResults(sorted_results, best_params, score_name=score_name, score_thr=score_thr, occ_method=occ_method)
        return final_results
    
    
def grid_search_test():
    train_dm, test_dm = get_usps_data(numbers = [0])
    results = grid_search(train_dm, prints = False, score_thr=0.9, score_name=nu_prec, nu_true=0.05)
    print(results.abv_thr_params)
    print(results.best_params)
    print(results.sorted_results.tail(10))
    
#grid_search_test()
    
    
        
    
    
    