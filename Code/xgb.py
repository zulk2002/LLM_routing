from DataSet import TrainingSet, TestSet
import xgboost as xgb
import numpy as np

class gb5:
    def __init__(self, depth = 5):
        self.model = xgb.XGBRegressor(max_depth=depth)

    def pretrain(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, context):
        return self.model.predict(context)
    
def evaluate_training_set(traing_set:TrainingSet, depth = 5):
    eval_idx, train_idx = training_set.split(seed=0,eval_rate=0.1)
    num_model = traing_set.scores.shape[1]

    model_list = [xgb.XGBRegressor(max_depth = depth) for _ in range(num_model)]
    for i in range(num_model):
        # print(i)
        model_list[i].fit(traing_set.features[train_idx,:],traing_set.scores[train_idx,i])

    res = []
    pred_list = [model_list[i].predict(traing_set.features[eval_idx,:]) for i in range(num_model)]
    res = np.argmax(pred_list,axis=0)
    
    return training_set.evaluate(np.array(res),idx=eval_idx)


if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]

    for i,name in enumerate(name_list):
        training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
        test_set = TestSet(f"./Demo/data/competition_data/{name}_test_pred.csv")
        training_set.read_feature(f"./Demo/data/features/{name}_train.csv")
        test_set.read_feature(f"./Demo/data/features/{name}_test_pred.csv")

        print(name,evaluate_training_set(training_set))