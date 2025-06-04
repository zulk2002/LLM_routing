from DataSet import TrainingSet, TestSet
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# FEATURE = "features_ch"
FEATURE = "features_mpnet"

def gmm(train_features, scores, eval_features, k, seed = 0, scale = False):
    new_tf = train_features.copy()
    new_ef = eval_features.copy()
    if scale:
        scaler = StandardScaler()
        new_tf = scaler.fit_transform(train_features)
        new_ef = scaler.transform(new_ef)
    model = GaussianMixture(n_components=k,random_state=int(time.time()),n_init=20,covariance_type="full")
    model.fit(new_tf)
    train_labels = model.predict(new_tf)
    eval_labels = model.predict(new_ef)
    best_models = [np.argmax(np.sum(scores[train_labels == i],axis=0)) for i in range(k)]
    return np.array(best_models)[eval_labels]

def evaluate_training_set(training_set:TrainingSet, seed = 0, k = 20):

    eval_idx, train_idx = training_set.split(seed=seed)
    eval_features, train_features = training_set.features[eval_idx,:], training_set.features[train_idx,:]
    # train_idx = np.arange(training_set.size)
    
    res = gmm(train_features,training_set.scores[train_idx,:],eval_features,k=k)
    res_greedy = [training_set.best_model(train_idx)] * len(eval_idx)
    return training_set.evaluate(np.array(res),idx=eval_idx) - training_set.evaluate(np.array(res_greedy),idx=eval_idx)

def try_k(name_list:list[str], k_option, times:int = 20, feature_name = ""):

    for i,name in enumerate(name_list):
        training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
        training_set.read_feature(f"./Demo/data/{feature_name}/{name}_train.csv")
        for k in k_option:
            tot = 0
            for _ in range(times):
                acc = evaluate_training_set(training_set,k=k,seed=int(time.time())+_)
                tot += acc
            print(f"dataset:{name}, k={k} average impove:{tot/times:.4f}")

def evaluate_k(name_list:list[str], k_list:list, times:int = 20, feature_name = ""):
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
    k_option = range(60,80)
    try_k(name_list[2:3],k_option,times=8,feature_name=feature_name)