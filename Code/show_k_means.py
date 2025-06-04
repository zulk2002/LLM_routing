from DataSet import TrainingSet, TestSet
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# FEATURE = "features_ch"
FEATURE = "features_mpnet"

def find_center(features, k, max_iters=100, seed=0, random = False):
    np.random.seed(seed)
    centers = features[np.random.choice(features.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        distances = np.linalg.norm(features[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        if random:
            eps = 0.5 - 0.45* _ / max_iters 
            # n = int(labels.shape[0] * eps)
            # selected = np.random.choice(labels.shape[0], size=n, replace=False)
            # new_labels = np.random.choice(k, size=n, replace=True)
            # labels[selected] = new_labels
            labels = np.where(np.random.uniform(0,1,labels.shape[0]) < eps, np.random.choice(k, size=labels.shape[0], replace=True), labels)
        new_centers = np.array([features[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centers, new_centers):
            # print(_)
            break
        centers = new_centers
    
    # distances_2 = np.sort(distances,axis=1)
    # print(np.sum(distances_2[:,0]**2))
    # print(distances_2[:10,:3])
    # dis_delta = distances_2[:,1] - distances_2[:,0]
    # plt.hist(dis_delta,bins=100)
    # y = np.sort(dis_delta)
    # plt.savefig("./tmp.png")
    return centers

def k_means(features, scores, k):
    centers = find_center(features,k = k, random = True)

    distances = np.linalg.norm(features[:, np.newaxis] - centers, axis=2)
    labels = np.argmin(distances, axis=1)
    best_models = [np.argmax(np.sum(scores[labels == i],axis=0)) for i in range(k)]
    return centers, np.array(best_models)

def solve_by_Kmeans(name, file_dictionary, k = 5):
    print(name)
    training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    test_set = TestSet(f"./Demo/data/competition_data/{name}_test_pred.csv")
    training_set.read_feature(f"./Demo/data/{FEATURE}/{name}_train.csv")
    test_set.read_feature(f"./Demo/data/{FEATURE}/{name}_test_pred.csv")

    centers, best_models = k_means(training_set.features,training_set.scores,k=k)
    distances = np.linalg.norm(test_set.features[:, np.newaxis] - centers, axis=2)
    labels = np.argmin(distances, axis=1)
    res = best_models[labels]
    res_names = training_set.dataframe.columns.to_numpy()[res+2]
    test_set.save_res(res_names,f"{file_dictionary}/{name}_test_pred.csv")
    return res_names

def evaluate_training_set(training_set:TrainingSet, seed = 0, k = 20):

    eval_idx, train_idx = training_set.split(seed=seed)
    eval_features, train_features = training_set.features[eval_idx,:], training_set.features[train_idx,:]
    # train_idx = np.arange(training_set.size)

    centers, best_models = k_means(train_features,training_set.scores[train_idx,:],k=k)

    eval_distances = np.linalg.norm(eval_features[:, np.newaxis] - centers, axis=2)
    labels = np.argmin(eval_distances, axis=1)
    res = best_models[labels]
    res_greedy = [training_set.best_model(train_idx)] * len(eval_idx)
    return training_set.evaluate(np.array(res),idx=eval_idx) - training_set.evaluate(np.array(res_greedy),idx=eval_idx)

def try_k(name_list:list[str], k_option, times:int = 20, feature_name = "features_p"):

    for i,name in enumerate(name_list):
        training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
        training_set.read_feature(f"./Demo/data/{feature_name}/{name}_train.csv")
        for k in k_option:
            tot = 0
            for _ in range(times):
                acc = evaluate_training_set(training_set,k=k,seed=int(time.time())+_)
                tot += acc
            print(f"dataset:{name}, k={k} average impove:{tot/times:.4f}")

def evaluate_k(name_list:list[str], k_list:list, times:int = 20, feature_name = "features_p"):
    for i,name in enumerate(name_list):
        acc_list = []
        training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
        training_set.read_feature(f"./Demo/data/{feature_name}/{name}_train.csv")
        for _ in range(times):
            acc = evaluate_training_set(training_set,k=k_list[i],seed=int(time.time())+_)
            acc_list.append(acc)
            # print(acc,end=" ")
        print(f"dataset:{name}, k={k_list[i]} average impove:{np.mean(acc_list):.4f}, std:{np.std(acc_list)}")

if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
    name = name_list[0]
    feature_name = "features_mpnet"
    # evaluate_k([name_list[0]],[8],times=20, feature_name=feature_name)
    # training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    # training_set.read_feature(f"./Demo/data/{feature_name}/{name}_train.csv")
    # evaluate_training_set(training_set,k=8)
    # k_list_o = [8,1,3,3,15,7,1]
    # k_list = [9,1,3,3,3,13,15]
    k_option = range(3,10)
    # evaluate_k([name_list[0]],[8],times=80,feature_name=feature_name)
    try_k([name_list[0]],k_option,times=80,feature_name=feature_name)
    # for i,name in enumerate(name_list):
    #     solve_by_Kmeans(name,f"./Demo/result_Kmeans",k=k_list_o[i])

    # solve_by_Kmeans(name_list[4],f"./Demo/result_Kmeans",k=k_list[4])