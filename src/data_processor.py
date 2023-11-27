""" 用于数据处理 """
import os
import re
import uuid
import pandas as pd
import numpy as np
from llama_index.finetuning import EmbeddingQAFinetuneDataset
import json
import argparse
from utils import logger
from tqdm import tqdm

parser = argparse.ArgumentParser(description="data preprocessor")

parser.add_argument("--process_func",default="adapter_data_process",type=str,
                    choices=["adapter_data_process","unsupervised_data_process","supervised_data_process"],
                    help="the unit function to preprocess data")
parser.add_argument("--combine_flag",action=True,default=True,
                    help="combine all data or not")
parser.add_argument("--instruction_flag",action=True,default=False,
                    help="use instruction or not,default False")
parser.add_argument("process_args",type=dict,default={},
                    help="the args of process_func,pass it as a str")
parser.add_argument("--return_amount",type=str,default="train",choices=['train','full','eval'],
                    help="the data collection to return,one of 'train','full','eval'")

from utils import INSTRUCTION_DICT
np.random.seed(123)


def data_reader( 
                path="./data",
                file_keyword='QA',
                id_cols=["文档id","切片id"],
                content_cols=["切片内容","Q"]
                    ):
    """读取file_keyword指定文件的数据，并取content_cols指定的列
    """
    data_files = [os.path.join(path,file) 
                for file in os.listdir(path) 
                if re.search(file_keyword, file)]
    data_list = []
    for file in data_files:
        data = pd.read_excel(file)[id_cols+content_cols]
        data_list.append(data)

    data_list = pd.concat(data_list)[content_cols]

    return data_list            


def data_dict_gen(
                    path_dir_list=("./data/",)*3,
                    file_keyword_list=("QA","QA","分片"),
                    id_cols_list=(["文档id"],)*3,
                    content_cols_list=(["切片内容","Q","切片id"],
                                       ["切片内容","A","切片id"],
                                       ["切片内容","摘要","切片id"]),
                    cols_map={"切片id":"doc_id","Q":"query",
                              "切片内容":"doc","A":"query",
                              "摘要":"query"
                              }
                                       ):
    """用于读取并做基本处理"""
    data_dict = {}
    key_list = ["content_query","content_answer","content_summary"]
    for key,path,file_keyword,id_cols,content_cols in zip(
                                                    key_list,
                                                    path_dir_list,
                                                    file_keyword_list,
                                                    id_cols_list,
                                                    content_cols_list
                                                    ):
        data_dict[key] = data_reader(path=path,
                                            file_keyword=file_keyword,
                                            id_cols=id_cols,
                                            content_cols=content_cols).rename(columns=cols_map)
    return data_dict

def add_instruction(data:pd.DataFrame,instruction:str=None):
    if instruction:
        data['query'] = instruction + ":" + data['query']
    return data


def adapter_data_process(data:pd.DataFrame):
    """用于生成llama-index的adapter tuning使用的数据格式
    nodes_dict, queries_dict, relevant_docs
    确保列名为：doc,query,doc_id
    """

    nodes_dict,queries_dict, relevant_docs = {},{},{}
    for idx, row in data.iterrows():
        question_id = str(uuid.uuid4())
        doc_id = row["doc_id"]
        doc = row["doc"]
        query = row["query"]
        nodes_dict[doc_id] = doc
        queries_dict[question_id] = query
        relevant_docs[question_id] = [doc_id] # 此处要求doc_id为一个list
    result = EmbeddingQAFinetuneDataset(
        queries=queries_dict,
        corpus=nodes_dict,
        relevant_docs=relevant_docs
    )
    # 按queries:queries_dict, corpus:node_dict, relevant_docs:relevant_docs的方式存为json
    # result.save_json("test.json",encode="utf-8",indent=4) 
    return result
            


def unsupervised_data_process(data: pd.DataFrame):
    pass


def supervised_data_process(data: pd.DataFrame,
                            output_dir="./data",
                            file_name="supervised_finetune_data.jsonl"):
    logger.info("process unit data")
    result = []
    temp = {}
    output_path = os.path.join(output_dir,file_name)
    file = open(output_path,mode="a",encoding="utf-8")
    total = data.shape[0]
    for i,row in tqdm(data.iterrows(),total=total):
        temp["query"] = row["query"]
        temp["pos"] = row["doc"]
        candidate = data.drop(i,inplace=False)["doc"]
        temp["neg"] = np.random.choice(candidate,1)
        json.dump(temp,file)
        result.append(temp)
    file.close()
    logger.info("process unit done.")
    return result


def shuffle_data(data: pd.DataFrame, return_amount:str="full",ratio:float=0.75):
    data = data.take(np.random.permutation(data.shape[0]))
    length = int(data.shape[0]*ratio)

    if return_amount == "full":
        return data 
    elif return_amount == "train":
        return data[:length]
    else:
        return data[length:]


def union_data_process(process_func=adapter_data_process,
                       combine_flag:bool=True,
                       instruction_flag:str=None,
                       process_args:dict={},
                       return_amount:str="train"):
    """用于不同格式数据的统一处理
       process_func: 处理数据格式的函数
       combine_flag: 是否合并返回
       instruction_flag:是否在query中使用instruction
       process_args: process_func的关键字参数
       return_amount:返回数量的量, full, train, eval,以3/4为训练集，1/4为测试集
    """
    logger.info("start to process data ...")
    data_df_dict = data_dict_gen()
    if instruction_flag:
        for key, data_df in data_df_dict.items():
            data_df_dict[key] = add_instruction(data=data_df,instruction=INSTRUCTION_DICT[key]) 
    
    if combine_flag:
        data_df = pd.concat(list(data_df_dict.values()))
        data_df = shuffle_data(data_df,return_amount=return_amount)
        result = process_func(data_df,**process_args)


    else:
        result = {}
        for key, value in data_df_dict.items():
            value = shuffle_data(value, return_amount=return_amount)
            temp = process_func(value,**process_args)
            result[key] = temp
    logger.info("process data done.")
    return result


if __name__ == "__main__":

    args = parser.parse_args()
    result = union_data_process(
        process_func=eval(args.process_func),
        combine_flag=args.combine_flag,
        instruction_flag=args.instruction_flag,
        process_args=eval(args.process_args), # 
        return_amount=args.return_amount
    )


