from imports import * 
from grid_search import * 
from data_loader import * 
from report_class import * 
from experiments import * 

def images_usps_experiment(numbers = [0], prints = False, gs_mode = True, occ_method = occ_osvm, im1 = [2, 2], im2 = [1, 5]):
    score_thr = 0.95
    
    experiment_id = ""
    for j in numbers: 
        experiment_id += str(j)
    
    if prints:
        print("USPS experiment with numbers = ")
        print(numbers)
    report = outlier_detection_usps(numbers = numbers, score_name=nu_rec_name, score_thr = score_thr, gs_mode = gs_mode, occ_method = occ_method)
    
    n_rows1, n_cols1  = im1[0], im1[1]
    n_rows2, n_cols2 = im2[0], im2[1]
    report.print_outliers(prints = False, 
                          n_rows1 = n_rows1,
                          n_cols1 = n_cols1, 
                          n_rows2 = n_rows2,
                          n_cols2 = n_cols2,
                          image_size = 2.0, 
                          id1 = experiment_id, 
                          id2 = experiment_id)
    
    return
    
    
def generate_usps_images():
    numbers_list = [[0], [2, 6], [0, 1, 2, 3, 4]]
    im1 = [1, 10]
    im2 = [1, 10]
    for numbers in numbers_list:
        images_usps_experiment(numbers = numbers, prints = True, gs_mode = True, occ_method = occ_osvm, im1= im1, im2 = im2)
    