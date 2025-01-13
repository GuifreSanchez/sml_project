from imports import * 


# data sets names: 
bc = "breast-cancer"
shuttle = "shuttle"
letter = "letter"
sat = "satellite"
pen_global = "pen-global"
annthyroid = "annthyroid"
aloi = "aloi"
datasets = [bc, shuttle, letter, sat, pen_global, annthyroid, aloi]

class DataManager:
    def __init__(self, data, labels, occ_labels):
        self.data = data
        self.labels = labels
        self.occ_labels = occ_labels
        self.check_occ_labels()
        self.nu = self.compute_nu()
        
    def check_occ_labels(self):
        for i in self.occ_labels: 
            if (i != -1 and i != +1):
                print("occ labels not equal to +-1")
                return
            
    def compute_nu(self):
        outliers = 0
        for l in self.occ_labels: 
            if l == -1:
                outliers += 1
        return outliers / len(self.occ_labels)
        

def load_data(name):
    path = 'datasets/dataverse_files/' + name + '-unsupervised-ad.csv'
    # Load CSV file into a DataFrame
    df = pd.read_csv(path, header = None)
    # Exclude the last row and convert the DataFrame to a NumPy array
    data = df.iloc[:, :-1].to_numpy()
    labeled_data = df.iloc[:].to_numpy()
    labels = labeled_data[:, -1]
    num_labels = np.array([-1 if label == 'o' else +1 for label in labels])
    n_outliers = len(num_labels[num_labels == -1])
    result = DataManager(data, num_labels, num_labels)
    return result


def add_outliers(data, outliers):
    n_outliers = len(outliers)
    labels = np.concatenate([np.ones(len(data)), -np.ones(len(outliers))], axis = 0)
    num_labels = labels
    full_data = np.concatenate([data, outliers], axis=0)
    labeled_data = np.c_[full_data, labels]
    result = DataManager(full_data, labels, labels)
    return result

# Here we make main data in first blob, and outliers in separated blob. 
def synth_2_modes(n_samples = 150, nu = 0.1, center1 = [-0.4, -0.4], center2 = [0.45, 0.45], std1 = 0.2, std2 = 0.1):
    n_outliers = int(n_samples * nu)
    n_inliers = n_samples - n_outliers  
    blobs_params_inliers = dict(random_state=0, n_samples=n_inliers, n_features=2)
    blobs_params_outliers = dict(random_state = 2, n_samples = n_outliers, n_features = 2)
    inliers = make_blobs(centers = [center1], cluster_std = std1, **blobs_params_inliers)[0]
    outliers = make_blobs(centers = [center2], cluster_std = std2, **blobs_params_outliers)[0]
    return add_outliers(inliers, outliers)

def synth_moons(n_samples = 150, nu = 0.1, noise = 0.05, scale = 4.0, shift = [0.5, 0.25], rs1 =0, rs2 = 3):
    n_outliers = int(n_samples * nu)
    n_inliers = n_samples - n_outliers
    inliers = scale * (make_moons(n_samples = n_inliers, noise = noise, random_state = rs1)[0]- np.array(shift))
    outliers = np.random.uniform(-scale, scale, size = (n_outliers, 2))
    return add_outliers(inliers, outliers)

def get_usps_data(numbers = [0]):
    path = 'datasets/usps.h5'
    with h5py.File(path, 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
    
    indices = np.array([])
    for n in numbers:
        new_indices = np.array(np.where(y_tr == n))[0, :]
        indices = np.concatenate([indices, new_indices], axis = 0)   
        
    indices = np.sort(np.array([int(i) for i in indices]))
    
    data = X_tr[indices]
    labels = y_tr[indices]
    occ_labels = np.ones(len(labels))
    
    test_data = X_te
    test_labels = y_te
    test_occ_labels = []
    
    for l in test_labels:
        occ_label_added = False
        for n in numbers: 
            if l == n:
                test_occ_labels.append(+1)
                occ_label_added = True
                break
        if occ_label_added is False:
            test_occ_labels.append(-1)
            
    train_dm = DataManager(data, labels, occ_labels)
    test_dm = DataManager(test_data, test_labels,test_occ_labels)
        
    return train_dm, test_dm 


def test_load_data():
    # breast_cancer_dm = load_data(bc)
    # synth1 = synth_2_modes()
    # synth2 = synth_moons()
    
    # print(breast_cancer_dm.data)
    # print(synth2.occ_labels)
    # print(synth2.data)
    
    # train_dm, test_dm  = get_usps_data(numbers = [0, 3, 8])
    # print(train_dm.labels)
    # for i in range(100):
    #     print(test_dm.labels[i], test_dm.occ_labels[i])
    return 