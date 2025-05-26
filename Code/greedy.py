from DataSet import TrainingSet, TestSet
import numpy as np

def evaluate_training_set(name):
    training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    training_set.read_feature(f"./Demo/data/features/{name}_train.csv")

    eval_idx, train_idx = training_set.split(seed=0)

    # order = np.argsort(training_set.scores.sum(axis=0))
    # print(order)

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