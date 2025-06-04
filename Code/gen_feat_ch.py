import os
import numpy as np
from sentence_transformers import SentenceTransformer
from DataSet import TrainingSet, TestSet
from transformers import AutoModel, AutoTokenizer
import torch

def gen_feat(model, name_list, file_dictionary):
    for i,name in enumerate(name_list):
        print(f"generating {name} ...")

        training_set = TrainingSet(f"./Demo/data/p_data/{name}_train.csv")
        sentences = training_set.questions
        embeddings = model.encode(sentences,device="cuda")
        np.savetxt(f"{file_dictionary}/{name}_train.csv", embeddings, delimiter=",")

        test_set = TestSet(f"./Demo/data/p_data/{name}_test.csv")
        sentences = test_set.questions
        embeddings = model.encode(sentences,device="cuda")
        np.savetxt(f"{file_dictionary}/{name}_test_pred.csv", embeddings, delimiter=",")

def encode_chinese(text, zh_tokenizer, zh_model):
    inputs = zh_tokenizer(text, return_tensors="pt", padding=True, truncation=True)#.to("cuda")
    # with torch.no_grad():
    outputs = zh_model(**inputs)
    return outputs.last_hidden_state[:, 0].detach().numpy()

def gen_feat_ch(tokenizer, model, name_list, file_dictionary):
    for i,name in enumerate(name_list):
        print(f"generating {name} ...")

        training_set = TrainingSet(f"./Demo/data/p_data/{name}_train.csv")
        sentences = training_set.questions
        embeddings = encode_chinese(sentences.tolist(),tokenizer,model)
        np.savetxt(f"{file_dictionary}/{name}_train.csv", embeddings, delimiter=",")

        test_set = TestSet(f"./Demo/data/p_data/{name}_test.csv")
        sentences = test_set.questions
        embeddings = encode_chinese(sentences.tolist(),tokenizer,model)
        np.savetxt(f"{file_dictionary}/{name}_test_pred.csv", embeddings, delimiter=",")

def proxy(address):
    os.environ['HTTP_PROXY'] = f'http://{address}'
    os.environ['HTTPS_PROXY'] = f'http://{address}'

def save_model(model:SentenceTransformer, file_dictionary):
    model.save(file_dictionary)
    print("Model saved successfully")

def save_model_ch(tokenizer, model, file_dictionary):
    tokenizer.save_pretrained(file_dictionary)
    model.save_pretrained(file_dictionary)

# proxy("10.19.130.93:7890")
zh_tokenizer = AutoTokenizer.from_pretrained("./Model/bge-base-zh")
zh_model = AutoModel.from_pretrained("./Model/bge-base-zh")#.to("cuda")

name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
for i,name in enumerate(name_list):
    gen_feat_ch(zh_tokenizer,zh_model,[name_list[0],name_list[2]],"./Demo/data/features_ch_p")
# gen_feat(model,name_list,"./Demo/data/features_p")

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