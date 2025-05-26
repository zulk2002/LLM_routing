from DataSet import TrainingSet, TestSet
import xgboost as xgb
import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None  # 存储模型参数（权重和截距）

    def train(self, X, y):
        """
        训练线性回归模型（最小二乘法闭式解）
        参数:
            X: 特征矩阵，形状 (n_samples, n_features)
            y: 目标值，形状 (n_samples,)
        """
        # 添加偏置项（截距）
        X_b = np.c_[np.ones(X.shape[0]), X]  # 形状变为 (n_samples, n_features+1)

        # 计算闭式解：θ = (X^T X)^(-1) X^T y
        self.coefficients = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        """
        预测新数据
        参数:
            X: 特征矩阵，形状 (n_samples, n_features)
        返回:
            预测值，形状 (n_samples,)
        """
        if self.coefficients is None:
            raise ValueError("Model not trained yet. Call `train()` first.")

        X_b = np.c_[np.ones(X.shape[0]), X]  # 添加偏置项
        return X_b @ self.coefficients  # 线性预测
    
def evaluate_training_set(traing_set:TrainingSet, k = 5):
    eval_idx, train_idx = training_set.split(seed=0)
    first_k = np.argsort(training_set.scores.sum(axis=0))[-k:]
    # print(first_5)
    model_list = [LinearRegression() for _ in range(k)]
    for i in range(k):
        model_list[i].train(traing_set.features[train_idx,:],traing_set.scores[train_idx,first_k[i]])

    res = []
    pred_list = [ model_list[i].predict(traing_set.features[eval_idx,:]) for i in range(k)]
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

        print(name,evaluate_training_set(training_set,k=2))