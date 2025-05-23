from DataSet import TrainingSet, TestSet
import numpy as np

def l2_distance(features,x):
    n = features.shape[0]
    m = x.shape[0]
    dist_map = np.zeros((n,m))
    # (x-y) ** 2 = x**2 + y**2 -2xy(dot)
    # print(dist_map)
    # print(np.sum(features**2, axis=1))
    dist_map += np.sum(features**2, axis=1).reshape(n,1)
    dist_map += np.sum(x**2, axis=1).reshape(1,m)
    dist_map -= 2*np.dot(features,x.T)
    return dist_map.T

def KNN(features, x, k):
    dist_map = l2_distance(features,x)
    nearest_k = np.argsort(dist_map)[:,:k]
    return nearest_k

def solve_by_KNN(name, file_dictionary, k = 5):
    training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    test_set = TestSet(f"./Demo/data/competition_data/{name}_test_pred.csv")
    training_set.read_feature(f"./Demo/data/features/{name}_train.csv")
    test_set.read_feature(f"./Demo/data/features/{name}_test_pred.csv")
    best_k = KNN(training_set.features, test_set.features, k)
    res = []
    for i in range(test_set.size):
        res.append(training_set.best_model(best_k[i]))
    res = np.array(res).astype(int)
    res_names = training_set.dataframe.columns.to_numpy()[res+2]
    test_set.save_res(res_names,f"{file_dictionary}/{name}_test_pred.csv")
    return res_names

def evaluate_training_set(name, k=5):
    training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    training_set.read_feature(f"./Demo/data/features/{name}_train.csv")
    # print(training_set.size)
    def split(N, eval_rate = 0.1):
        np.random.seed(0)
        N_eval = int(N * eval_rate)
        idx_shuffle = np.arange(N)
        np.random.shuffle(idx_shuffle)
        return idx_shuffle[:N_eval], idx_shuffle[N_eval:]

    eval_idx, train_idx = split(training_set.size)
    # train_idx = np.arange(training_set.size)

    
    best_k = KNN(training_set.features[train_idx,:], training_set.features[eval_idx,:], k)
    res = []
    for i in range(eval_idx.shape[0]):
        res.append(training_set.best_model(train_idx[best_k[i]]))
    return training_set.evaluate(np.array(res),idx=eval_idx)
    # training_set.evaluate([training_set.best_model()] * training_set.size)

if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]

    k_list = [1101,201,1001,101,1501,1201,501]
    for i,name in enumerate(name_list):
        # print(name,evaluate_training_set(name, k = k_list[i]))
        solve_by_KNN(name,f"./Demo/result_KNN")
    # x = np.array([-1,0,0,0,1,0]).reshape(3,2)
    # y = np.array([-1,-1,-1,1,1,-1,1,1]).reshape(4,2)
    # print(KNN(y,x,2))