from DataSet import TrainingSet, TestSet
import xgboost as xgb
import numpy as np
import time

FEATURE = "features_p"
    
def evaluate_training_set(traing_set:TrainingSet, depth = 20, k = 5, seed = 0):
    eval_idx, train_idx = training_set.split(seed=seed)
    first_k = np.argsort(training_set.scores.sum(axis=0))[-k:]
    model_list = [xgb.XGBRegressor(max_depth = depth) for _ in range(k)]
    for i in range(k):
        model_list[i].fit(traing_set.features[train_idx,:],traing_set.scores[train_idx,first_k[i]])

    res = []
    pred_list = [ model_list[i].predict(traing_set.features[eval_idx,:]) for i in range(k)]
    res = first_k[np.argmax(pred_list,axis=0)]
    res_greedy = [training_set.best_model(train_idx)] * len(eval_idx)
    return training_set.evaluate(np.array(res),idx=eval_idx) - training_set.evaluate(np.array(res_greedy),idx=eval_idx)

def evaluate(training_set, times = 20):
    acc = 0
    for _ in range(times):
        acc+=evaluate_training_set(training_set,seed=int(time.time()+_))
    print(f"avg:{acc/times:.4f}")

if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
    for i,name in enumerate(name_list[:2]):
        training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
        test_set = TestSet(f"./Demo/data/competition_data/{name}_test_pred.csv")
        training_set.read_feature(f"./Demo/data/{FEATURE}/{name}_train.csv",tag_path=f"./Demo/data/p_data/{name}_train_tag.csv")
        # test_set.read_feature(f"./Demo/data/{FEATURE}/{name}_test_pred.csv",tag_path=f"./Demo/data/p_data/{name}_test_tag.csv")
        # print(name,evaluate_training_set(training_set,k=5))
        evaluate(training_set,times=5)