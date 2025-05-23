from DataSet import TrainingSet, TestSet
import numpy as np

def evaluate_training_set(name):
    training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    training_set.read_feature(f"./Demo/data/features/{name}_train.csv")

    def split(N, eval_rate = 0.1):
        np.random.seed(0)
        N_eval = int(N * eval_rate)
        idx_shuffle = np.arange(N)
        np.random.shuffle(idx_shuffle)
        return idx_shuffle[:N_eval], idx_shuffle[N_eval:]

    eval_idx, train_idx = split(training_set.size)

    res = []
    for i in range(eval_idx.shape[0]):
        res.append(training_set.best_model(train_idx))
    return training_set.evaluate(np.array(res),idx=eval_idx)

if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
    # acc_list = []
    for i,name in enumerate(name_list):
        # res_name = solve_by_KNN(name,"./Demo/result_KNN")
        # print()
        print(name,evaluate_training_set(name))
    # print(np.mean(acc_list))