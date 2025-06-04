from DataSet import TrainingSet, TestSet
import numpy as np
import timeit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def k_means(t_features, scores, e_featrues, k,seed = 0, scale = False):
    new_tf = t_features.copy()
    new_ef = e_featrues.copy()
    print(new_ef.shape)
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

    print(f"best: {best_models}")
    return best_models[e_labels], best_models[labels]

def draw_pic(training_set:TrainingSet, seed = 0, k = 20):
    from tsne import get_emb, plot

    eval_idx, train_idx = training_set.split(seed=seed)
    embeddings = get_emb(training_set.features)

    # eval_features, train_features = embeddings[eval_idx,:], embeddings[train_idx,:]
    eval_features, train_features = training_set.features[eval_idx,:], training_set.features[train_idx,:]

    e_res,t_res = k_means(train_features,training_set.scores[train_idx,:],eval_features,k=k)
    t_colors = training_set.get_score_by_ans(t_res,train_idx)
    t_colors = np.where(t_colors,"blue","red")
    e_colors = training_set.get_score_by_ans(e_res,eval_idx)
    e_colors = np.where(e_colors,"blue","red")
    plot(embeddings[train_idx,:],"./plot_t2.png",colors=t_colors)
    plot(embeddings[eval_idx,:],"./plot_e2.png",colors=e_colors)
    tot_e = np.vstack([embeddings[train_idx,:],embeddings[eval_idx,:]])
    tot_c = np.hstack([t_colors,e_colors])
    plot(tot_e,"./plot_tot.png",tot_c)

    res_greedy = [training_set.best_model(train_idx)] * len(eval_idx)
    return training_set.evaluate(np.array(e_res),idx=eval_idx) - training_set.evaluate(np.array(res_greedy),idx=eval_idx)


if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
    name = name_list[0]
    feature_name = "features_mpnet"
    training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    training_set.read_feature(f"./Demo/data/{feature_name}/{name}_train.csv")
    k = 4
    print(draw_pic(training_set,k=k))