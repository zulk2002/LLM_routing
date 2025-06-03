from DataSet import TrainingSet, TestSet
import xgboost as xgb
import numpy as np
import json
import itertools

FEATURE = "features_mpnet"

def index_split(N, seed=0, ratio = 0.8):
        np.random.seed(seed)
        N_before = int(N * ratio)
        idx_shuffle = np.arange(N)
        np.random.shuffle(idx_shuffle)
        return idx_shuffle[:N_before], idx_shuffle[N_before:]
    
BEST_GREEDY_LMS = {
    "aclue": "glm_4_plus",
    "arc_c": "glm_4_plus",
    "cmmlu": "qwen25_72b_instruct",
    "hotpot_qa": "gpt_4o",
    "math": "deepseek_coder",
    "mmlu": "llama31_405b_instruct",
    "squad": "llama31_405b_instruct",
}

class PromptWithLMFeaturesDataset():
    def __init__(self, features, LM_features, labels):
        self.data = self.concatenate_prompt_LM_feature(features, LM_features) # shape = ((D * K), (prompt_feature_dim + LM_feature_dim))
        self.labels = labels # shape = (D * K)
        # print(f"Dataset size: {self.data.shape}, Labels size: {self.labels.shape}")
        # print(labels[:10])

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def concatenate_prompt_LM_feature(self, prompt_features, LM_features):
        # prompt_feature: D * prompt_feature_dim
        # LM_feature: K * LM_feature_dim
        # return (D * K) + (prompt_feature_dim + LM_feature_dim)
        # print(prompt_feature.shape)
        # print(LM_feature.shape)
        D = prompt_features.shape[0]
        K = LM_features.shape[0]
        prompt_features = np.repeat(prompt_features, K, axis=0)  # (D * K) * prompt_feature_dim
        LM_features = np.tile(LM_features, (D, 1))  # (D * K) * LM_feature_dim
        # print(LM_feature.shape)
        return np.concatenate((prompt_features, LM_features), axis=1)  # (D * K) * (prompt_feature_dim + LM_feature_dim)

class gb5:
    def __init__(self, depth = 5):
        self.model = xgb.XGBRegressor(max_depth=depth)

    def pretrain(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, context):
        return self.model.predict(context)
    
def evaluate_training_set(training_set:TrainingSet, All_LM_features, depth = 7, k = 5):
    eval_idx, train_idx = training_set.split(seed=0)
    first_k = np.argsort(-training_set.scores.sum(axis=0))[:k]
    # 提取模型特征
    LM_list = [list(All_LM_features[0].keys())[i] for i in first_k]
    LM_features = np.array([[feature[LM] for LM in LM_list] for feature in All_LM_features]).T # K * ori_LM_feature_dim
        
    # 对模型进行 one-hot 编码
    LM_one_hot = np.eye(len(LM_list), dtype=np.float32)  # K * K
    LM_features = np.concatenate((LM_features, LM_one_hot), axis=1)  # K * (LM_feature_dim + K)
    print(LM_features)
    data_size = training_set.features.shape[0]
    train_index, val_index = index_split(data_size, seed=0, ratio=0.8)
    labels = [training_set.scores[i, j] for i, j in itertools.product(train_index, first_k)] 
    train_dataset = PromptWithLMFeaturesDataset(training_set.features[train_index], LM_features, labels)
    # print(first_5)
    # model_list = [xgb.XGBRegressor(max_depth = depth) for _ in range(k)]
    model = xgb.XGBRFRegressor(max_depth = depth)
    model.fit(train_dataset.data, train_dataset.labels)
    
    res = []
    for i in val_index:
        rewards = []
        for j in range(len(first_k)):
            prompt_input_feature = training_set.features[i].reshape(1, -1)  # 1 * prompt_feature_dim
            LM_input_feature = LM_features[j].reshape(1, -1)  # 1 * (LM_feature_dim + K)
            val_input = np.concatenate((prompt_input_feature, LM_input_feature), axis=1)  # 1 * (prompt_feature_dim + LM_feature_dim + K)
            prediction = model.predict(val_input)
            rewards.append(prediction)
        action = first_k[np.argmax(rewards)]
        res.append(action)
    
    # for i in range(k):
        # model_list[i].fit(traing_set.features[train_idx,:],traing_set.scores[train_idx,first_k[i]])

    # res = []
    # pred_list = [ model_list[i].predict(traing_set.features[eval_idx,:]) for i in range(k)]
    # res = first_k[np.argmax(pred_list,axis=0)]
    res_greedy = [training_set.best_model(train_idx)] * len(eval_idx)
    return training_set.evaluate(np.array(res),idx=eval_idx) - training_set.evaluate(np.array(res_greedy),idx=eval_idx)


if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
    for i,name in enumerate(name_list):
        training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
        test_set = TestSet(f"./Demo/data/competition_data/{name}_test_pred.csv")
        training_set.read_feature(f"./Demo/data/{FEATURE}/{name}_train.csv")
        test_set.read_feature(f"./Demo/data/{FEATURE}/{name}_test_pred.csv")
        model_features_for_all_datasets = json.load(open("./Code/model_features.json","r"))
        model_features = [model_feature[name] for model_feature in model_features_for_all_datasets]
        print(name,evaluate_training_set(training_set,model_features, k=5))