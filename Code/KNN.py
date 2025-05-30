from DataSet import TrainingSet, TestSet
import numpy as np
import time

FEATURE = "features_p"

def l2_distance(features,x):
    n = features.shape[0]
    m = x.shape[0]
    dist_map = np.zeros((n,m))
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
    training_set.read_feature(f"./Demo/data/{FEATURE}/{name}_train.csv")
    test_set.read_feature(f"./Demo/data/{FEATURE}/{name}_test_pred.csv")
    best_k = KNN(training_set.features, test_set.features, k)
    res = []
    for i in range(test_set.size):
        res.append(training_set.best_model(best_k[i]))
    res = np.array(res).astype(int)
    res_names = training_set.dataframe.columns.to_numpy()[res+2]
    test_set.save_res(res_names,f"{file_dictionary}/{name}_test_pred.csv")
    return res_names

def evaluate_training_set(name, k=5, seed = 0):
    training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    training_set.read_feature(f"./Demo/data/{FEATURE}/{name}_train.csv")
    # print(training_set.size)

    eval_idx, train_idx = training_set.split(seed=seed)
    # train_idx = np.arange(training_set.size)

    
    best_k = KNN(training_set.features[train_idx,:], training_set.features[eval_idx,:], k)
    res = [ training_set.best_model(train_idx[best_k[i]]) for i in range(eval_idx.shape[0])]
    res_greedy = [training_set.best_model(train_idx)] * len(eval_idx)
    return training_set.evaluate(np.array(res),idx=eval_idx) - training_set.evaluate(np.array(res_greedy),idx=eval_idx)
    # training_set.evaluate([training_set.best_model()] * training_set.size)

def try_k(name_list:list[str], k_option:list, times:int = 20):
    for i,name in enumerate(name_list):
        for k in k_option:
            tot = 0
            for _ in range(times):
                acc = evaluate_training_set(name,k=k,seed=int(time.time())+i)
                tot += acc
            print(f"dataset:{name}, k={k} average impove:{tot:.4f}")

if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
    k_list = [300,250,1051,320,1060,800,700]
    # for i,name in enumerate(name_list):
    #     solve_by_KNN(name,"./Demo/result_KNN",k=k_list[i])
    try_k(name_list,[50,100,300,800,1000])