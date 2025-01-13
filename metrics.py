from imports import * 

def custom_recall(y_true, y_pred):
    return custom_metric(y_true, y_pred, metric = 'recall')

def custom_precision(y_true, y_pred):
    return custom_metric(y_true, y_pred, metric = 'precision')

def custom_nu_prec(y_true, y_pred, nu):
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

f1 = 'f1'
rec = 'recall'
prec = 'precision'
nu_prec = 'nu_precision' 
pr_auc = 'pr_auc'

score_names = [f1, rec, prec, nu_prec, pr_auc]

def select_score(score_name = 'f1'):
    if score_name == f1:
        return custom_f1
    elif score_name == rec:
        return custom_recall
    elif score_name == prec: 
        return custom_precision
    elif score_name == nu_prec:
        return custom_nu_prec
    elif score_name == pr_auc:
        return custom_pr_auc
    else:
        print("selected score not implemented")
        return 