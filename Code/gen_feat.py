
import numpy as np
from sentence_transformers import SentenceTransformer
from DataSet import TrainingSet, TestSet



model = SentenceTransformer("./model").to("cuda")

name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
for i,name in enumerate(name_list):
    print(name)
    training_set = TrainingSet(f"./data/competition_data/{name}_train.csv")
    sentences = training_set.questions
    embeddings = model.encode(sentences,device="cuda")
    np.savetxt(f"data/features/{name}_train.csv", embeddings, delimiter=",")
    test_set = TestSet(f"./data/competition_data/{name}_test_pred.csv")
    sentences = test_set.questions
    embeddings = model.encode(sentences,device="cuda")
    np.savetxt(f"data/features/{name}_test_pred.csv", embeddings, delimiter=",")



# # The sentences to encode
# sentences = [
#     """关于：basic_ancient_chinese
# 问题：下列定义是许慎为“形声”所下的是（）
# 选项：
# A. ⽐类合谊，以⻅指撝
# B. 建类⼀⾸，同意相受
# C. 以事为名，取譬相成
# D. 视⽽可识，察⽽⻅意
# """,
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]

# # 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode(sentences)
# print(embeddings.shape)
# similarities = model.similarity(embeddings, embeddings)
# print(similarities)
# # tensor([[1.0000, 0.6660, 0.1046],
# #         [0.6660, 1.0000, 0.1411],
# #         [0.1046, 0.1411, 1.0000]])