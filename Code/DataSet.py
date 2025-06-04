import numpy as np
import pandas as pd
from typing import Callable

class TrainingSet:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataframe = pd.read_csv(file_path)
        self.size = self.dataframe.shape[0]
        self.questions = self.dataframe.iloc[:,1].to_numpy() # could design a datasructure if needed
        self.scores = self.dataframe.iloc[:,2:22].to_numpy()
        self.opt_score = np.sum(np.max(self.scores,axis=1))

    def read_feature(self,file_path,tag_path=None):
        self.features = np.loadtxt(file_path, delimiter=",")
        if tag_path is not None:
            add_tag = self.read_tags(tag_path)
            self.features = np.hstack([self.features, add_tag])
    
    def read_tags(self,file_path):
        tags = np.loadtxt(file_path, delimiter=",")
        def one_hot(arr:np.ndarray):
            arr = arr.astype(int)
            n_classes = np.max(arr)+1
            one_hot = np.zeros((len(arr),n_classes))
            one_hot[np.arange(len(arr)),arr] = 1
            return one_hot
        add_tag = one_hot(tags) if tags.ndim == 1 else np.hstack([one_hot(tags[:,0]),one_hot(tags[:,1])])
        return add_tag

    def best_model(self, idx=None):
        idx = idx if idx is not None else np.arange(self.size)
        return self.scores[idx].sum(axis=0).argmax()
    
    def best_k(self,k,idx=None):
        idx = idx if idx is not None else np.arange(self.size)
        rank = np.argsort(self.scores[idx].sum(axis=0))
        return rank[-k:]
    
    def difference(self,score_table:np.ndarray):
        total_score:np.ndarray = score_table.sum(axis=0)
        order = np.argsort(total_score)
        best = order[-1]
        second_best = order[-2]
        print(f"{best} {total_score[best]:.0f}:{total_score[second_best]:.0f} {second_best}")
        print(np.sum(np.maximum(score_table[:,second_best]-score_table[:,best],0)))
        print(np.sum(np.maximum(score_table[:,best]-score_table[:,second_best],0)))
    
    def compare(self, model_1, model_2, idx = None):
        idx = idx if idx is not None else np.arange(self.size)
        score_slice = self.scores[idx]
        best_score = np.max(score_slice,axis=1)
        conditions = [best_score == score_slice[:,model_1], best_score == score_slice[:,model_2],
                      best_score >  score_slice[:,model_1], best_score >  score_slice[:,model_2]]
        print(np.sum(conditions[0] & conditions[1]))
        a,b = np.sum(conditions[1] & conditions[0]),np.sum(conditions[1] & conditions[2])
        c,d = np.sum(conditions[3] & conditions[0]),np.sum(conditions[3] & conditions[2])
        print(f"\t{model_1} yes\t{model_1} no")
        print(f"{model_2} yes\t{a}\t{b}")
        print(f"{model_2} no\t{c}\t{d}")

    def percent(self,score_table:np.ndarray):
        opt = self.opt_score
        total_score:np.ndarray = score_table.sum(axis=0)
        order = np.argsort(total_score)
        for i in range(20):
            res = np.sum(np.max(score_table[:,order[-i:]],axis=1))
            print(f"{i}:{res/opt*100:.2f}%,{res/self.size*100:.2f}%")

    def get_opt_score(self, idx):
        return np.sum(np.max(self.scores[idx,:],axis=1))

    def evaluate(self, ans, idx=None, absolute=False):
        idx = idx if idx is not None else np.arange(self.size)
        performance = np.sum(self.get_score_by_ans(ans,idx))
        opt_idx = self.get_opt_score(idx)
        # print(f"Opitmal score:{opt_idx}")
        # print(f"Your score:{performance}")
        # print(f"Percentage:{performance/opt_idx*100:.2f}")
        return performance/opt_idx if absolute == False else performance

    def split(self, seed=0, eval_rate = 0.2, drop_zero=0):
        np.random.seed(seed)
        N_eval = int(self.size * eval_rate)
        idx_shuffle = np.arange(self.size)
        np.random.shuffle(idx_shuffle)
        t_idx = idx_shuffle[N_eval:]
        if drop_zero > 0:
            best_k_idx = self.best_k(drop_zero,t_idx)
            # print(best_k_idx)
            non_zero_idx = np.where(np.sum(self.scores[np.ix_(t_idx,best_k_idx)],axis=1)>0)[0]
            # print(len(t_idx))
            t_idx=t_idx[non_zero_idx]
            # print(len(t_idx))
            # exit(0)
        return idx_shuffle[:N_eval], t_idx

    def get_score_by_ans(self, ans, idx=None):
        idx = idx if idx is not None else np.arange(self.size)
        score_table = self.scores[idx,:]
        return score_table[np.arange(len(ans)),ans]


class TestSet:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataframe = pd.read_csv(file_path)
        self.size = self.dataframe.shape[0]
        self.questions = self.dataframe.iloc[:,1].to_numpy()

    def read_feature(self,file_path,tag_path=None):
        self.features = np.loadtxt(file_path, delimiter=",")
        if tag_path is not None:
            add_tag = self.read_tags(tag_path)
            self.features = np.hstack([self.features, add_tag])

    
    def read_tags(self,file_path):
        tags = np.loadtxt(file_path, delimiter=",")
        def one_hot(arr:np.ndarray):
            arr = arr.astype(int)
            n_classes = np.max(arr)+1
            one_hot = np.zeros((len(arr),n_classes))
            one_hot[np.arange(len(arr)),arr] = 1
            return one_hot
        add_tag = one_hot(tags) if tags.ndim == 1 else np.hstack([one_hot(tags[:,0]),one_hot(tags[:,1])])
        return add_tag

    def save_res(self,res_list, file_path):
        res = self.dataframe.copy()
        res["pred"] = res_list
        res.to_csv(file_path,index=False)



if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
    feature_name_list = ["features","features_mpnet","features_p"]
    # for i,name in enumerate(name_list):
    #     # print(name)
    #     tmp = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    #     for feature_name in feature_name_list:
    #         tmp.read_feature(f"./Demo/data/{feature_name}/{name}_train.csv")
    #         from tsne import draw
    #         draw(tmp.features,f"./Demo/picture/{name}_{feature_name}.png")
    

    name = name_list[0]
    feature_name = feature_name_list[1]
    tmp = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    tmp.read_feature(f"./Demo/data/{feature_name}/{name}_train.csv")
    model_id = 17
    colors = ["blue" if score == 1 else "red" for score in tmp.scores[:,model_id]]
    from tsne import draw
    draw(tmp.features,f"./Demo/picture/{name}_{feature_name}.png",colors)