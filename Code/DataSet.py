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
    
    def difference(self,score_table:np.ndarray):
        total_score:np.ndarray = score_table.sum(axis=0)
        order = np.argsort(total_score)
        best = order[-1]
        second_best = order[-2]
        print(f"{best} {total_score[best]:.0f}:{total_score[second_best]:.0f} {second_best}")
        print(np.sum(np.maximum(score_table[:,second_best]-score_table[:,best],0)))
        print(np.sum(np.maximum(score_table[:,best]-score_table[:,second_best],0)))
    
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
        score_table = self.scores[idx,:]
        performance = np.sum(score_table[np.arange(len(ans)),ans])
        opt_idx = self.get_opt_score(idx)
        # print(f"Opitmal score:{opt_idx}")
        # print(f"Your score:{performance}")
        # print(f"Percentage:{performance/opt_idx*100:.2f}")
        return performance/opt_idx if absolute == False else performance
    
    def split(self, seed=0, eval_rate = 0.2):
        np.random.seed(seed)
        N_eval = int(self.size * eval_rate)
        idx_shuffle = np.arange(self.size)
        np.random.shuffle(idx_shuffle)
        return idx_shuffle[:N_eval], idx_shuffle[N_eval:]
        


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
    for i,name in enumerate(name_list):
        # print(name)
        tmp = TestSet(f"./Demo/data/p_data/{name}_test.csv")
        # tmp.read_feature()