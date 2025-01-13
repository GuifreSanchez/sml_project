from imports import * 

def custom_recall(y_true, y_pred):
    return custom_metric(y_true, y_pred, metric = 'recall')

def custom_precision(y_true, y_pred):
    return custom_metric(y_true, y_pred, metric = 'precision')


prec_w = 0.5
nu_w = 1.0 - prec_w
def custom_nu_prec(y_true, y_pred, nu_true = 0.05):
    precision = custom_precision(y_true, y_pred)
    nu_score = custom_nu_score(y_true, y_pred, nu_true = nu_true)
    return prec_w * precision + nu_w * nu_score

def custom_param_nu_prec(nu_true =0.05):
    def func(y_true, y_pred):
        return custom_nu_prec(y_true, y_pred, nu_true = nu_true)
    return func

rec_w = 0.75
nu_w = 1.0 - rec_w
def custom_nu_rec(y_true, y_pred, nu_true = 0.05):
    recall = custom_recall(y_true, y_pred)
    nu_score = custom_nu_score(y_true, y_pred, nu_true = nu_true)
    return rec_w * recall + nu_w * nu_score

def custom_param_nu_rec(nu_true =0.05):
    def func(y_true, y_pred):
        return custom_nu_rec(y_true, y_pred, nu_true = nu_true)
    return func

def custom_nu_score(y_true, y_pred, nu_true = 0.05):
    tp, tn, fp, fn = custom_metric(y_true, y_pred, metric = 'info')
    N = tn + fn
    P = tp + fp
    nu_pred = N / (N + P)
    nu_dist_inv = 2 / (1 + (nu_pred - nu_true)**2) - 1
    return nu_dist_inv
    

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
    return result

f1_name = 'f1'
rec_name = 'recall'
prec_name = 'precision'
nu_prec_name = 'nu_precision' 
nu_rec_name = 'nu_recall'
pr_auc_name = 'pr_auc'
nu_name = 'nu'

score_names = [f1_name, rec_name, prec_name, nu_prec_name, pr_auc_name, nu_name]

def select_score(score_name = 'f1', nu_true = 0.05):
    if score_name == f1_name:
        return custom_f1
    elif score_name == rec_name:
        return custom_recall
    elif score_name == prec_name: 
        return custom_precision
    elif score_name == nu_prec_name:
        return custom_param_nu_prec(nu_true = nu_true)
    elif score_name == nu_rec_name:
        return custom_param_nu_rec(nu_true = nu_true)
    elif score_name == pr_auc_name:
        return custom_pr_auc
    elif score_name == nu_name: 
        return custom_nu_score
    else:
        print("selected score not implemented")
        return 