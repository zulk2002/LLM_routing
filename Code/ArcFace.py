from DataSet import TrainingSet, TestSet
import numpy as np
import torch
import torch.nn as nn
import json
import math
import torch.nn.functional as F

class ArcFaceNet(nn.Module):
    """ArcFace层实现"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, input, label):
        normalized_input = F.normalize(input)
        normalized_weight = F.normalize(self.weight)
        cosine = F.linear(normalized_input, normalized_weight)
        theta = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        adjusted_theta = theta * self.cos_m - torch.sqrt(1.0 - cosine**2) * self.sin_m

        final_theta = torch.where(cosine > self.th, adjusted_theta, theta)
        final_theta = final_theta * self.s
        return final_theta

    def predict(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine * self.s

class ArcFaceTrainer:
    def __init__(self, in_features, out_features, lr=0.01):
        self.model = ArcFaceNet(in_features, out_features)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def loss(self, outputs, labels):
        return nn.BCEWithLogitsLoss()(outputs, labels)
    
    def train(self, features, labels, epochs=500, batch_size=128):
        self.model.train()
        self.optimizer.zero_grad()
        n_samples = len(features)
        for epoch in range(epochs):
            indices = torch.randperm(n_samples)
            X = features[indices]
            y = labels[indices]
            for i in range(0, n_samples, batch_size):
                batch_features = X[i:i+batch_size]
                batch_labels = y[i:i+batch_size]
                batch_features = torch.tensor(batch_features, dtype=torch.float32)
                batch_labels = torch.tensor(batch_labels, dtype=torch.float32)
                outputs = self.model(batch_features, batch_labels)
                loss = self.loss(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
        
    def predict(self, features):
        self.model.eval()
        with torch.no_grad():
            features = torch.tensor(features, dtype=torch.float32)
            outputs = self.model.predict(features)
        return outputs.numpy()


def evaluate_training_set(traing_set:TrainingSet, model_fearures, k = 5):
    eval_idx, train_idx = training_set.split(seed=0)
    first_k = np.argsort(training_set.scores.sum(axis=0))[-k:]
    # print(first_k)
    feature_size = traing_set.features.shape[1]
    trainer = ArcFaceTrainer(feature_size, k)
    labels = np.array([traing_set.scores[train_idx,first_k[i]] for i in range(k)]).T
    trainer.train(traing_set.features[train_idx,:], labels)
    # for i in range(k):
    #     chosen_LM = model.LM_lists[first_k[i]]
    #     model.train(traing_set.features[train_idx,:], chosen_LM, traing_set.scores[train_idx,first_k[i]])
    res = []
    pred_list = trainer.predict(traing_set.features[eval_idx,:])
    print(pred_list)
    # pred_list = [model.predict(traing_set.features[eval_idx,:], model.LM_lists[first_k[i]]) for i in range(k)]
    # print(np.array(pred_list).T)
    GT = [traing_set.scores[eval_idx,first_k[i]] for i in range(k)]
    print(np.array(GT).T)
    # prob = np.array(softmax(pred_list)).T
    # print(prob)
    # res = [np.random.choice(first_k, p=prob[i]) for i in range(len(eval_idx))]
    res = first_k[np.argmax(pred_list,axis=1)]
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