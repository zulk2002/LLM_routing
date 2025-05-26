from DataSet import TrainingSet, TestSet
import numpy as np
import time

def find_center(features, k, max_iters=1000, seed=0):
    np.random.seed(seed)
    centers = features[np.random.choice(features.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        distances = np.linalg.norm(features[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([features[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    
    return centers

def k_means(features, scores, k):
    centers = find_center(features,k = k)

    distances = np.linalg.norm(features[:, np.newaxis] - centers, axis=2)
    labels = np.argmin(distances, axis=1)
    best_models = [np.argmax(np.sum(scores[labels == i],axis=0)) for i in range(k)]
    return centers, np.array(best_models)

def solve_by_Kmeans(name, file_dictionary, k = 5):
    training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    test_set = TestSet(f"./Demo/data/competition_data/{name}_test_pred.csv")
    training_set.read_feature(f"./Demo/data/features/{name}_train.csv")
    test_set.read_feature(f"./Demo/data/features/{name}_test_pred.csv")

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

if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
    k_option = [14,15]
    k_list = [3,3,3,3,3,13,2]

    # tot_list = []
    # for i,name in enumerate(name_list):
    #     if i != 5:
    #         continue
    #     for k in k_option:
    #         tot = 0
    #         for _ in range(20):
    #             training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    #             training_set.read_feature(f"./Demo/data/features/{name}_train.csv")
    #             tot += evaluate_training_set(training_set,k=k,seed=int(time.time()))
    #         print(k,name,tot)

    # tot_list = []
    # for i,name in enumerate(name_list):
    #     if i != 0:
    #         continue
    #     res_list = []
    #     for k in k_option:
    #         training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    #         training_set.read_feature(f"./Demo/data/features/{name}_train.csv")
    #         acc = evaluate_training_set(training_set,k=k)
    #         print(k,name,acc)
    #         res_list.append(np.round(acc,2))
    #     tot_list.append(res_list)
    # print(tot_list)

    for i,name in enumerate(name_list):
        solve_by_Kmeans(name,f"./Demo/res_Kmeans",k=k_list[i])