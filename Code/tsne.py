from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def get_emb(features):
    # 假设你的数据存储在变量 `features` 中 (n_samples, 768)
    # 1. 标准化数据 (非常重要!)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 2. 创建并拟合 t-SNE 模型
    tsne = TSNE(
        n_components=2,      # 降到2维
        perplexity=70,       # 尝试调整这个值！ 30 是常见起点
        max_iter=1000,         # 默认1000，不够可增加
        learning_rate=200,   # 默认200，根据结果调整
        random_state=42,     # 固定随机种子保证可复现
        init='pca',          # 推荐使用PCA初始化
        verbose=1          # 打印进度
    )
    embeddings = tsne.fit_transform(features_scaled)
    return embeddings

def plot(embeddings, fig_name:str="", colors=None):
    plt.figure(figsize=(10, 8))
    if colors is not None:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.4,c=colors)  # alpha 控制点透明度
    else:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.4)  # alpha 控制点透明度
    plt.title('t-SNE Visualization of 768-D Feature Data')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
    if len(fig_name) > 0:
        plt.savefig(fig_name)
    plt.cla()
    plt.close()

def draw(features, fig_name:str="", colors=None):
    embeddings = get_emb(features)
    print(embeddings.shape)
    plot(embeddings,fig_name=fig_name,colors=colors)