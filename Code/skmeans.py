from DataSet import TrainingSet, TestSet
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def k_means(t_features, scores, e_featrues, k,seed = 0, scale = False, draw = False):
    new_tf = t_features.copy()
    new_ef = e_featrues.copy()
    if scale:
        scaler = StandardScaler()
        new_tf = scaler.fit_transform(t_features)
        new_ef = scaler.transform(e_featrues)
    model = KMeans(n_clusters=k,random_state=seed,n_init=20)
    model.fit(new_tf)
    labels = model.labels_
    e_labels = model.predict(new_ef)
    # print(e_labels[:10])
    best_models = np.array([np.argmax(np.sum(scores[labels == i],axis=0)) for i in range(k)])

    if draw:
        return best_models[e_labels], best_models[labels]

    return best_models[e_labels]

def solve_by_Kmeans(name, file_dictionary, k = 5, feature_name = ""):
    print(name)
    training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    test_set = TestSet(f"./Demo/data/competition_data/{name}_test_pred.csv")
    training_set.read_feature(f"./Demo/data/{feature_name}/{name}_train.csv")
    test_set.read_feature(f"./Demo/data/{feature_name}/{name}_test_pred.csv")

    centers, best_models = k_means(training_set.features,training_set.scores,k=k,scale=True)
    distances = np.linalg.norm(test_set.features[:, np.newaxis] - centers, axis=2)
    labels = np.argmin(distances, axis=1)
    res = best_models[labels]
    res_names = training_set.dataframe.columns.to_numpy()[res+2]
    test_set.save_res(res_names,f"{file_dictionary}/{name}_test_pred.csv")
    return res_names

def draw_pic(training_set:TrainingSet, seed = 0, k = 20):
    eval_idx, train_idx = training_set.split(seed=seed)
    eval_features, train_features = training_set.features[eval_idx,:], training_set.features[train_idx,:]
    e_res,t_res = k_means(train_features,training_set.scores[train_idx,:],eval_features,k=k,draw=True)
    
    from tsne import get_emb, plot
    embeddings = get_emb(training_set.features)
    t_colors = training_set.get_score_by_ans(t_res,train_idx)
    t_colors = np.where(t_colors,"blue","red")
    e_colors = training_set.get_score_by_ans(e_res,eval_idx)
    e_colors = np.where(e_colors,"blue","red")
    plot(embeddings[train_idx,:],"./plot_t.png",colors=t_colors)
    plot(embeddings[eval_idx,:],"./plot_e.png",colors=e_colors)

def evaluate_training_set(training_set:TrainingSet, seed = 0, k = 20):
    eval_idx, train_idx = training_set.split(seed=seed,drop_zero=0)
    eval_features, train_features = training_set.features[eval_idx,:], training_set.features[train_idx,:]
    res = k_means(train_features,training_set.scores[train_idx,:],eval_features,k=k,scale=False)

    res_greedy = [training_set.best_model(train_idx)] * len(eval_idx)
    return (training_set.evaluate(np.array(res),idx=eval_idx,absolute=True) - training_set.evaluate(np.array(res_greedy),idx=eval_idx,absolute=True))/len(eval_idx)

def try_k(name_list:list[str], k_option, times:int = 20, feature_name = ""):
    res_list = []
    for i,name in enumerate(name_list):
        training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
        training_set.read_feature(f"./Demo/data/{feature_name}/{name}_train.csv")
        acc_list = []
        for k in k_option:
            tot = 0
            for _ in range(times):
                acc = evaluate_training_set(training_set,k=k,seed=_)
                # acc = evaluate_training_set(training_set,k=k,seed=int(time.time())+_+k)
                tot += acc
            acc_list.append(tot/times);
            print(f"dataset:{name}, k={k} average impove:{tot/times:.4f}")
        res = {
            "name": name,
            "acc":acc_list,
            "k":k_option
        }
        res_list.append(res)
    return res_list

def evaluate_k(name_list:list[str], k_list:list, times:int = 20, feature_name = ""):
    for i,name in enumerate(name_list):
        acc_list = []
        training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
        training_set.read_feature(f"./Demo/data/{feature_name}/{name}_train.csv")
        for _ in range(times):
            acc = evaluate_training_set(training_set,k=k_list[i],seed=_)
            acc_list.append(acc)
            # print(acc,end=" ")
        print(f"dataset:{name}, k={k_list[i]} average impove:{np.mean(acc_list):.4f}, std:{np.std(acc_list)}")

def plot_res(res_list:list[dict],file_name):
    for res in res_list:
        plt.plot(res["k"],res["acc"],label=res["name"])
    plt.legend()
    plt.savefig(file_name)
        

if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
    name = name_list[0]
    feature_name = "features_mpnet"
    # evaluate_k([name_list[0]],[5],times=80, feature_name=feature_name)
    # training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    # training_set.read_feature(f"./Demo/data/{feature_name}/{name}_train.csv")
    # e_idx, t_idx = training_set.split(seed=0,drop_zero=True)
    # res = k_means(training_set.features[t_idx,:],training_set.scores[t_idx,:],training_set.features[e_idx],k=8)
    # print(training_set.evaluate(res,e_idx,absolute=True))
    # print(len(e_idx))
    # evaluate_training_set(training_set,k=8)
    # k_list_o = [8,1,3,3,15,7,1]
    # k_list = [9,1,3,3,3,13,15]
    k_option = range(2,51,2)
    res_list = try_k(name_list,k_option,times=50,feature_name=feature_name)
    plot_res(res_list,"./drop0.png")
    # for i,name in enumerate(name_list):
    #     solve_by_Kmeans(name,f"./Demo/result_Kmeans",k=k_list_o[i])

    # solve_by_Kmeans(name_list[4],f"./Demo/result_Kmeans",k=k_list[4])

    ############## draw picture #############
    # training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    # training_set.read_feature(f"./Demo/data/{feature_name}/{name}_train.csv")
    # k = 9
    # draw_pic(training_set,k=k)
    # print(evaluate_training_set(training_set,k=k))