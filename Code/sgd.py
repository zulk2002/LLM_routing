from DataSet import TrainingSet, TestSet
import numpy as np
import torch
import torch.nn as nn
import json

class Model(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)

class LinearRegression:
    def __init__(self, feature_size, LM_features):
        self.model = Model(feature_size, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.alpha = 0.3
        self.LM_features = LM_features
        self.LM_lists = list(LM_features[0].keys())
    
    def loss(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2) + self.alpha * torch.mean(self.model.fc.weight ** 2)
    
    def concatenate(self, X, a):
        LM_feature_input = np.repeat(np.array([LM_feature[a] for LM_feature in self.LM_features]), len(X), axis=0)
        return np.concatenate((X, np.array(LM_feature_input).reshape(-1,1)), axis=1)

    def train(self, X, a, y, epochs=500, batch_size=128):
        X = self.concatenate(X, a)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).reshape(-1, 1)
        n_samples = len(X)
        
        for epoch in range(epochs):
            indices = torch.randperm(n_samples)
            X = X[indices]
            y = y[indices]
            
            for i in range(0, n_samples, batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                y_pred = self.model(batch_X)
                loss = self.loss(y_pred, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def predict(self, X, a):
        X = self.concatenate(X, a)
        X = torch.FloatTensor(X)
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy().flatten()
        
    
def evaluate_training_set(traing_set:TrainingSet, model_fearures, k = 5):
    eval_idx, train_idx = training_set.split(seed=0)
    first_k = np.argsort(training_set.scores.sum(axis=0))[-k:]
    # print(first_k)
    feature_size = traing_set.features.shape[1] + len(model_features)
    model = LinearRegression(feature_size, model_fearures)
    for i in range(k):
        chosen_LM = model.LM_lists[first_k[i]]
        model.train(traing_set.features[train_idx,:], chosen_LM, traing_set.scores[train_idx,first_k[i]])
    res = []
    pred_list = [model.predict(traing_set.features[eval_idx,:], model.LM_lists[first_k[i]]) for i in range(k)]
    res = first_k[np.argmax(pred_list,axis=0)]
    res_greedy = [training_set.best_model(train_idx)] * len(eval_idx)
    return training_set.evaluate(np.array(res),idx=eval_idx) - training_set.evaluate(np.array(res_greedy),idx=eval_idx)


if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]

    for i,name in enumerate(name_list):
        training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
        test_set = TestSet(f"./Demo/data/competition_data/{name}_test_pred.csv")
        training_set.read_feature(f"./Demo/data/features/{name}_train.csv")
        test_set.read_feature(f"./Demo/data/features/{name}_test_pred.csv")
        model_features_for_all_datasets = json.load(open("./Code/model_features.json","r"))
        model_features = [model_feature[name] for model_feature in model_features_for_all_datasets]
        print(name,evaluate_training_set(training_set, model_features, k=5))