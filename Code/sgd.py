from DataSet import TrainingSet, TestSet
import numpy as np
import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from tqdm import tqdm
import itertools
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



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

class PromptWithLMFeaturesDataset(Dataset):
    def __init__(self, features, LM_features, labels):
        self.data = torch.tensor(self.concatenate_prompt_LM_feature(features, LM_features), dtype=torch.float32) # shape = ((D * K), (prompt_feature_dim + LM_feature_dim))
        self.labels = torch.tensor(labels, dtype=torch.float32) # shape = (D * K)
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

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)

class MLPModel(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc3 = nn.Linear(512, output_size)
        self.fc2 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        self.dropout2 = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        self.batchnorm1 = nn.BatchNorm1d(512)  # Batch normalization layer
        self.batchnorm2 = nn.BatchNorm1d(512)

        self.fc1.weight.data.normal_(0, 0.01)  # Initialize weights
        self.fc2.weight.data.normal_(0, 0.01)  # Initialize weights
        self.fc3.weight.data.normal_(0, 0.01)  # Initialize weights
        self.fc1.bias.data.fill_(0)  # Initialize biases
        self.fc2.bias.data.fill_(0)  # Initialize biases
        self.fc3.bias.data.fill_(0)  # Initialize biases

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = F.sigmoid(self.fc3(x))
        return x.squeeze()
    

class SGDTrainer:
    def __init__(self,
                    training_set:TrainingSet,
                    All_LM_features,
                    k=10,
                    lr=1e-3,
                    alpha=20.0, 
                    batch_size=128,
                    epochs=1000
                 ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据加载器初始化
        self.__init_dataloader__(training_set, All_LM_features, k, batch_size=batch_size)
        
        # 模型初始化
        self.model = MLPModel(self.feature_size, 1).to(self.device)
        
        # 优化器初始化
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-7)
        
        self.alpha = alpha
        self.margin = 0.5
        self.beta = 0.0
        
        self.log = {
            "train_loss": [],
            "val_rewards": [],
            "val_epochs": [],
            "val_predict_rewards": [],
            "lr": [],
            "val_loss": [],
        }
    
    def __init_dataloader__(self, training_set:TrainingSet, All_LM_features, k=5, batch_size=128):
        #training_set: D * prompt_feature_dim
        data_size = training_set.features.shape[0]

        # 只考虑前k个比较厉害的模型
        self.first_k = np.argsort(-training_set.scores.sum(axis=0))[:k]
        # print(first_k)

        # 提取模型特征
        self.LM_list = [list(All_LM_features[0].keys())[i] for i in self.first_k]
        self.LM_features = np.array([[feature[LM] for LM in self.LM_list] for feature in All_LM_features]).T # K * ori_LM_feature_dim
        
        # 对模型进行 one-hot 编码
        LM_one_hot = np.eye(len(self.LM_list), dtype=np.float32)  # K * K
        self.LM_features = np.concatenate((self.LM_features, LM_one_hot), axis=1)  # K * (LM_feature_dim + K)
        print(self.LM_features)

        # 构建 dataloader
        train_index, self.val_index = index_split(data_size, seed=0, ratio=0.8)
        
        # 构建 label
        labels = [training_set.scores[i, j] for i, j in itertools.product(train_index, self.first_k)] 
        
        train_dataset = PromptWithLMFeaturesDataset(training_set.features[train_index], self.LM_features, labels)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_prompt_features = training_set.features[self.val_index]
        self.val_labels = training_set.scores[self.val_index, :]  # (D_val, 20)
        self.feature_size = train_dataset.data.shape[1]  # prompt_feature_dim + LM_feature_dim + K
        
    def action_embedding(self, action):
        # action: str
        # return: LM_feature_dim
        action_index = self.LM_list.index(action)
        return self.LM_features[action_index, :].reshape(1, -1)  # (1, LM_feature_dim)

    # def used_LM_features(self, used_LM_index):
    #     # print(used_LM_index)
    #     used_LM_list = [self.LM_lists[i] for i in used_LM_index]
    #     return np.array([[feature[LM] for LM in used_LM_list] for feature in self.LM_features]).T
    
    def loss(self, y_pred, y_true):
        weight = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        num = 0
        for parameter in self.model.parameters():
            weight += torch.mean(parameter ** 2)
            num += 1
        weight /= num
        # pos_mask = (y_true > 0.5)
        # neg_mask = (y_true <= 0.5)
        # ranking_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        # valid_pairs = 0
        
        # for i in range(len(y_pred)):
        #     for j in range(i + 1, len(y_pred)):
        #         if pos_mask[i] and neg_mask[j]:
        #             valid_pairs += 1
        #             ranking_loss += F.relu(self.margin - (y_pred[i][0] - y_pred[j][0]))
        #         elif pos_mask[j] and neg_mask[i]:
        #             valid_pairs += 1
        #             ranking_loss += F.relu(self.margin - (y_pred[j][0] - y_pred[i][0]))
        # if valid_pairs > 0:
        #     ranking_loss /= valid_pairs
        # else:
        #     ranking_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        return torch.mean((y_pred - y_true) ** 2) + self.alpha * weight
    # + ranking_loss * self.beta
    
    # def concatenate(self, X, a):
    #     LM_feature_input = np.repeat(np.array([LM_feature[a] for LM_feature in self.LM_features]), len(X), axis=0)
    #     return np.concatenate((X, np.array(LM_feature_input).reshape(-1,1)), axis=1)
    
    def train_one_step(self, batch_X, batch_y):
        self.model.train()
        y_pred = self.model(batch_X)
        loss = self.loss(y_pred, batch_y)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def validate(self, epoch=None, print_loss=False, description=None):
        self.model.eval()
        with torch.no_grad():
            actions, pre_rewards = self.take_action(self.val_prompt_features)
            acutal_rewards = [label[self.first_k[self.LM_list.index(action)]] for label, action in zip(self.val_labels, actions)]
            if epoch is not None:
                self.log["val_epochs"].append(epoch)
                self.log["val_predict_rewards"].append(np.mean(pre_rewards))
                self.log["val_rewards"].append(np.mean(acutal_rewards))
                self.log["val_loss"].append(self.loss(torch.tensor(pre_rewards, dtype=torch.float32).to(self.device), torch.tensor(acutal_rewards, dtype=torch.float32).to(self.device)).item())
            if print_loss:
                if description is None:
                    description = f"Validation at epoch {epoch}"
                print(f"{description}: {np.mean(acutal_rewards):.4f}, Predicted: {np.mean(pre_rewards):.4f}")
        return acutal_rewards
    
    def fit(self, val_freq=10):
        self.validate(print_loss=True, description="Initial Validation")
        with tqdm(total=self.epochs, desc="Training Progress") as pbar:
            pbar.set_postfix({"loss": 0})
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_X, batch_y in self.train_dataloader:
                    loss = self.train_one_step(batch_X.to(self.device), batch_y.to(self.device))
                    total_loss += loss
                self.scheduler.step()
                avg_loss = total_loss / len(self.train_dataloader)
                self.log["train_loss"].append(avg_loss)
                self.log["lr"].append(self.optimizer.param_groups[0]['lr'])
                pbar.set_postfix({"loss": avg_loss})
                if (epoch + 1) == self.epochs or (epoch + 1) % val_freq == 0:
                    self.validate(epoch=epoch + 1)
                pbar.update(1)  
        # print("Model parameters:")
        # for name, param in self.model.named_parameters():
        #     print(f"Name: {name}")
        #     print(f"Shape: {param.shape}")
        #     print(f"Values: {param.data}\n")  
        return

    # def concatenate(self, X, a):
    #     # X: DS,  prompt_feature_dim
    #     # a: DS or 1
    #     # if 
    #     pass
    
    def take_action(self, X):
        # X: DS,  prompt_feature_dim
        # return: DS, 1
        
        # (K, LM_feature_dim, A)
        rewards = []
        for action in self.LM_list:
            action_embedding = np.repeat(self.action_embedding(action), X.shape[0], axis=0)  # (DS, LM_feature_dim, A)
            input = np.concatenate((X, action_embedding), axis=1)  # (DS, prompt_feature_dim + LM_feature_dim)
            input = torch.tensor(input, dtype=torch.float32).to(self.device)
            reward = self.model(input).to("cpu").numpy().flatten()  # (DS,)
            rewards.append(reward)
        rewards = np.array(rewards).T  # (DS, K)
        action_index = np.argmax(rewards, axis=1)  # (DS,)
        index = np.random.choice(np.arange(rewards.shape[0]))  # (DS,)
        # print(index, rewards[index, :], self.val_labels[index, self.first_k], action_index[index], self.LM_list[action_index[index]])
        rewards = np.max(rewards, axis=1)  # (DS,)
        actions = [self.LM_list[i] for i in action_index]  # (DS,)
        return actions, rewards  # (DS,)
        
        
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)


if __name__ == "__main__":
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
    
    
    over_all_fig, over_all_ax = plt.subplots(len(name_list), 4, figsize=(20, 35))
    for i,name in enumerate(name_list):
        training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
        test_set = TestSet(f"./Demo/data/competition_data/{name}_test_pred.csv")
        training_set.read_feature(f"./Demo/data/features_mpnet/{name}_train.csv")
        test_set.read_feature(f"./Demo/data/features_mpnet/{name}_test_pred.csv")
        model_features_for_all_datasets = json.load(open("./Code/model_features.json","r"))
        model_features = [model_feature[name] for model_feature in model_features_for_all_datasets]
        print(f"Training on {name}.....")
        # print(name,evaluate_training_set(name, training_set, model_features, k=5))
        # training(training_set, model_features, k=5)
        trainer = SGDTrainer(training_set, model_features, k=10)
        trainer.fit()
        
        # 计算贪心解
        greedy_LM = BEST_GREEDY_LMS[name]
        greedy_reward = np.array(training_set.scores[trainer.val_index, trainer.first_k[trainer.LM_list.index(greedy_LM)]]).mean()
        
        
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].plot(trainer.log["train_loss"])
        ax[0].set_title(f"Training Loss on {name}")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[1].plot(trainer.log["val_epochs"], trainer.log["val_rewards"], label="Linear Regression")
        ax[1].axhline(greedy_reward, color='red', linestyle='--', label=f"Greedy Reward: {greedy_reward:.4f}")
        ax[1].legend()
        ax[1].set_title(f"Validation Average Rewards on {name}")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Average Rewards")
        # ax[2].plot(trainer.log["val_epochs"], trainer.log["val_predict_rewards"], label="Predicted Rewards")
        # ax[2].plot(trainer.log["val_epochs"], trainer.log["val_rewards"], label="Actual Rewards")
        # ax[2].legend()
        # ax[2].set_title(f"Predicted vs Actual Rewards on {name}")
        # ax[2].set_xlabel("Epochs")
        # ax[2].set_ylabel("Rewards")
        ax[2].plot(trainer.log["val_epochs"], trainer.log["val_loss"])
        ax[2].set_title(f"Validation Loss on {name}")
        ax[2].set_xlabel("Epochs")
        ax[2].set_ylabel("Loss")
        ax[3].plot(trainer.log["lr"])
        ax[3].set_title(f"Learning Rate on {name}")
        ax[3].set_xlabel("Epochs")
        ax[3].set_ylabel("Learning Rate")
        fig.savefig(f"./temp/{name}.png")
        plt.close(fig)
        
        over_all_ax[i, 0].plot(trainer.log["train_loss"])
        over_all_ax[i, 0].set_title(f"Training Loss on {name}")
        over_all_ax[i, 0].set_xlabel("Epochs")
        over_all_ax[i, 0].set_ylabel("Loss")
        over_all_ax[i, 1].plot(trainer.log["val_epochs"], trainer.log["val_rewards"], label="Regression")
        over_all_ax[i, 1].axhline(greedy_reward, color='red', linestyle='--', label=f"Greedy Reward: {greedy_reward:.4f}")
        over_all_ax[i, 1].legend()
        over_all_ax[i, 1].set_title(f"Validation Average Rewards on {name}")
        over_all_ax[i, 1].set_xlabel("Epochs")
        over_all_ax[i, 1].set_ylabel("Average Rewards")
        # over_all_ax[i, 2].plot(trainer.log["val_epochs"], trainer.log["val_predict_rewards"], label="Predicted Rewards")
        # over_all_ax[i, 2].plot(trainer.log["val_epochs"], trainer.log["val_rewards"], label="Actual Rewards")
        # over_all_ax[i, 2].legend()
        # over_all_ax[i, 2].set_title(f"Predicted vs Actual Rewards on {name}")
        # over_all_ax[i, 2].set_xlabel("Epochs")
        # over_all_ax[i, 2].set_ylabel("Rewards")
        over_all_ax[i, 2].plot(trainer.log["val_epochs"], trainer.log["val_loss"])
        over_all_ax[i, 2].set_title(f"Validation Loss on {name}")
        over_all_ax[i, 2].set_xlabel("Epochs")
        over_all_ax[i, 2].set_ylabel("Loss")
        over_all_ax[i, 3].plot(trainer.log["lr"])
        over_all_ax[i, 3].set_title(f"Learning Rate on {name}")
        over_all_ax[i, 3].set_xlabel("Epochs")
        over_all_ax[i, 3].set_ylabel("Learning Rate")

        
    over_all_fig.savefig("./temp/mlp_over_all.png", bbox_inches='tight')
    plt.close(over_all_fig)