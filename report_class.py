from imports import * 
from data_loader import * 
from metrics import * 

class Report:
    def __init__(self, train_dm : DataManager, test_dm : DataManager, occ_method, occ_parameters, y_pred, nu_true):
        self.train_dm = train_dm
        self.test_dm = test_dm
        self.occ_method = occ_method
        self.occ_parameters = occ_parameters
        self.y_pred = y_pred
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
        return [f1, prec, rec, nu_prec, nu_rec]