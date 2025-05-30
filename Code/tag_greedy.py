from DataSet import TrainingSet, TestSet
import numpy as np
import time

def get_tag(questions:np.ndarray):
    tag_list = []
    for question in questions:
        tag = question[:question.find("\n")]
        if not tag in tag_list:
            tag_list.append(tag)
    return tag_list

def get_idx(q1:np.ndarray, q2:np.ndarray,tag_list:list[str]):
    idx_list = {}
    for tag in tag_list:
        idx1 = [tag in txt for txt in q1]
        idx1 = np.where(np.array(idx1) == True)[0]
        idx2 = [tag in txt for txt in q2]
        idx2 = np.where(np.array(idx2) == True)[0]
        idx_list[tag] = (idx1, idx2)
    return idx_list


def evaluate_training_set(training_set:TrainingSet, seed = 0):
    eval_idx, train_idx = training_set.split(seed=seed)
    tag_list = get_tag(training_set.questions[train_idx])
    
    idx_list = {" ": (np.arange(train_idx.shape[0]),np.arange(eval_idx.shape[0]))} if len(tag_list)>100 else \
        get_idx(training_set.questions[train_idx],training_set.questions[eval_idx],tag_list)

    res = np.zeros_like(eval_idx)
    for (tr,ev) in idx_list.values():
        best = training_set.best_model(train_idx[tr])
        res[ev] = best

    res_greedy = [training_set.best_model(train_idx)] * len(eval_idx)
    return training_set.evaluate(np.array(res),idx=eval_idx) - training_set.evaluate(np.array(res_greedy),idx=eval_idx)

def get_tag_2(training_set:TrainingSet):
    tag_list = []
    for question in training_set.questions:
        tag_list.append(question[:question.find("\n")])
    return np.array(tag_list)

def evaluate_training_set_2(training_set:TrainingSet, seed = 0):
    tag_list = get_tag_2(training_set)
    eval_idx, train_idx = training_set.split(seed=seed)
    res = []
    for e_id in eval_idx:
        p = []
        for t_id in train_idx:
            if tag_list[e_id] == tag_list[t_id]:
                p.append(t_id)
        # print(p)
        # exit(0)
        res.append(training_set.best_model(np.array(p)))
    
    res_greedy = [training_set.best_model(train_idx)] * len(eval_idx)
    return training_set.evaluate(np.array(res),idx=eval_idx) - training_set.evaluate(np.array(res_greedy),idx=eval_idx)
        


if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
    # acc_list = []
    for i,name in enumerate(name_list):
        tot = 0
        for _ in range(1):
            training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
            training_set.read_feature(f"./Demo/data/features/{name}_train.csv")

            tag_list = np.array(get_tag(training_set.questions))
            tag_list = tag_list if len(tag_list)<100 else np.array([" "])
            print(name,tag_list)

            # acc_improve = evaluate_training_set(training_set,seed=int(time.time())+i)
            # acc_improve = evaluate_training_set(training_set)
            # tot += acc_improve
        # print(tot)
    # print(np.mean(acc_list))
    # name = name_list[0]
    # training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    # training_set.read_feature(f"./Demo/data/features/{name}_train.csv")
    # print(evaluate_training_set(training_set))
    # print(evaluate_training_set_2(training_set))