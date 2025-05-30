from DataSet import TrainingSet, TestSet
import pandas
import numpy as np
import time

def get_tag(questions:np.ndarray):
    tag_list = []
    for question in questions:
        tag = question[:question.find("\n")]
        if not tag in tag_list:
            tag_list.append(tag)
    print(tag_list)
    exit(0)
    return tag_list

def remove_line(questions:np.ndarray):
    result = np.array(["\n".join(s.split('\n')[1:]) for s in questions])
    return result

def get_tag_val(tag_list:list[str],questions:np.ndarray):
    tag_val = np.zeros_like(questions)
    for i,tag in enumerate(tag_list):
        idx = [tag in txt for txt in questions]
        idx = np.where(np.array(idx) == True)[0]
        tag_val[idx] = i
    tag_val = tag_val.astype(int)
    return tag_val

def split_tag(train_df:pandas.DataFrame, test_df:pandas.DataFrame):
    train_questions, test_questions = train_df.iloc[:,1].to_numpy(),test_df.iloc[:,1].to_numpy()
    train_val, test_val = np.empty((0,train_questions.shape[0])), np.empty((0,test_questions.shape[0]))
    while(True):
        tag_list = get_tag( np.concatenate([train_questions, test_questions]) )
        if len(tag_list) > 100:
            break
        train_val = np.vstack([train_val,get_tag_val(tag_list,train_questions)])
        test_val = np.vstack([test_val,get_tag_val(tag_list,test_questions)])
        train_questions, test_questions = remove_line(train_questions), remove_line(test_questions)
    if train_val.shape[0] == 0 :
        train_val = np.vstack([train_val,np.zeros_like(train_questions)])
        test_val = np.vstack([test_val,np.zeros_like(test_questions)])

    train_val = np.transpose(train_val)
    test_val = np.transpose(test_val)
    return train_questions,train_val,test_questions,test_val

def save_data(file_dictionary:str, name:str, df:pandas.DataFrame, tag:np.ndarray):
    df.to_csv(f"{file_dictionary}/{name}.csv",index=False)
    np.savetxt(f"{file_dictionary}/{name}_tag.csv", tag, delimiter=",",fmt="%d")

def process(name_list:list[str]):
    for name in name_list:
        training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
        test_set = TestSet(f"./Demo/data/competition_data/{name}_test_pred.csv")

        train_questions,train_val,test_questions,test_val = split_tag(training_set.dataframe,test_set.dataframe)
        train_df, test_df = training_set.dataframe.copy(), test_set.dataframe.copy()
        train_df["question"], test_df["question"] = train_questions, test_questions
        save_data("./Demo/data/p_data",f"{name}_train",train_df,train_val)
        save_data("./Demo/data/p_data",f"{name}_test",test_df,test_val)

def test():
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]

    name = "math"
    training_set = TrainingSet(f"./Demo/data/competition_data/{name}_train.csv")
    test_set = TestSet(f"./Demo/data/competition_data/{name}_test_pred.csv")

    train_questions,train_val,test_questions,test_val = split_tag(training_set.dataframe,test_set.dataframe)
    train_df, test_df = training_set.dataframe.copy(), test_set.dataframe.copy()
    train_df["question"], test_df["question"] = train_questions, test_questions
    save_data("./Demo/data/p_data",f"{name}_train",train_df,train_val)
    save_data("./Demo/data/p_data",f"{name}_test",test_df,test_val)
    exit(0)

if __name__ == "__main__":
    test()
    name_list = ["aclue","arc_c","cmmlu","hotpot_qa","math","mmlu","squad"]
    process(name_list)
    