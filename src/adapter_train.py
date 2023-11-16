""" 用于基于非对称标注数据的adapter微调，基于llama-index实现"""
import os
from utils import logger
from typing import Union
from llama_index.finetuning import (SentenceTransformersFinetuneEngine,
                                    EmbeddingAdapterFinetuneEngine)
from llama_index.embeddings import (resolve_embed_model,
                                    AdapterEmbeddingModel,
                                    HuggingFaceEmbedding,)
from transformers import AutoModelForCausalLM,AutoTokenizer
from llama_index.llms import HuggingFaceLLM
from llama_index.schema import TextNode
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings.adapter_utils import TwoLayerNN
from llama_index.evaluation import RetrieverEvaluator
from data_processor import union_data_process, adapter_data_process
import pandas as pd
import asyncio

os.environ["TRANSFORMERS_OFFLINE"] = '1'
EMBED_MODEL_PATH = "local:/home/star/models/m3e-base"
LLM_MODEL_PATH = "/home/star/models/Qwen-14B-Chat-Int4/"


def li_train_finetune(model_name:str=EMBED_MODEL_PATH,
                  model_output_path:str="model_output/bge_large_cn_adapter_linear", # 保存最终的checkpoint的路径
                  model_checkpoint_path:str=None, # 保存中间checkpoint文件的路径
                  engine_class:str="adapter", # adapter,sentence_transformers
                  adapter_model:str=None, # TwoLayerNN, Linear
                  use_instruction:bool=False
                  ):
    logger.info("llama-index adapter fine-tuning start...")

    data = union_data_process(process_func=adapter_data_process,
                              combine_flag=True,
                              instruction_flag=use_instruction,
                              return_amount="train"
                              )
    logger.info("data procession done..")
    # resolve_embed_model(model_name),model_name必须是local: repo_id的格式
    base_embed_model = resolve_embed_model(model_name)
    if adapter_model is not None:
        adapter_model = eval(adapter_model)
        in_features_dim = base_embed_model._model.config.hidden_size
        adapter_model_ins = adapter_model(
        in_features=in_features_dim,  # input dimension
        hidden_features=in_features_dim*3,  # hidden dimension
        out_features=in_features_dim,  # output dimension
        bias=True,
        add_residual=True,
            )
    else:
        adapter_model_ins = None    

    if engine_class == "adapter":
        finetune_engine = EmbeddingAdapterFinetuneEngine(
            data,
            base_embed_model,
            model_output_path=model_output_path,
            model_checkpoint_path=model_checkpoint_path,
            adapter_model=adapter_model_ins,
            epochs=25,
            verbose=True,
        )
    else:
        finetune_engine = SentenceTransformersFinetuneEngine(
            data,
            model_id=model_name,
            model_output_path=model_output_path
        )
    logger.info("fine-tuning start...")
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model(
    adapter_cls=adapter_model
    )
    logger.info("llama-index adapter fine-tuning done.")
    return embed_model


def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)
    full_df = pd.DataFrame(metric_dicts)
    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    metric_df = pd.DataFrame(
        {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}
    )
    return metric_df    
    # return metric_df


def eval_embedding(embed_model,model,tokenizer,nodes,data,flag="finetuned"):
    llm = HuggingFaceLLM(model=model,tokenizer=tokenizer)
    service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
    vector_index = VectorStoreIndex(nodes, service_context=service_context)
    retriever = vector_index.as_retriever(similarity_top_k=2)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
    )
    eval_results = asyncio.run(retriever_evaluator.aevaluate_dataset(data))
    metric_df = display_results("top-2 eval", eval_results)
    print(f"----------------performance of {flag}----------------")
    print(metric_df)


def li_eval_finetune(finetune_class:str="adapter", # adapter,sentence_transformers
                  adapter_class: str="Linear", # None for linear, "TwoLayerNN"
                  model_name:str=EMBED_MODEL_PATH,
                  model_output_path:str="model_output/bge_large_cn_adapter_linear",
                  use_instruction:bool=False
                  ):
    """ 以异步的方式按照llama-index的retriever模式评估模型的表现
    para@finetune_class: 
    """
    data = union_data_process(process_func=adapter_data_process,
                              combine_flag=True,
                              instruction_flag=use_instruction,
                              return_amount="eval"
                              )
    
    if finetune_class == "adapter":
        base_embed_model = resolve_embed_model(model_name)
        adapter_class = eval(adapter_class) if adapter_class =="TwoLayerNN" else None
        embed_model = AdapterEmbeddingModel(
            base_embed_model=base_embed_model,
            adapter_path=model_output_path,
            adapter_cls=adapter_class
        )
    
    else:
        embed_model = HuggingFaceEmbedding(model_name=model_output_path)
    nodes = [TextNode(text=value, id_=key) for key,value in data.corpus.items()]

    # 针对qwen-14-int4,注意要在config.json中的quantization_config中加入disable_exllama:true
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH,trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH,trust_remote_code=True)

    eval_embedding(embed_model,model,tokenizer,nodes,data,flag="finetuned"),
    eval_embedding(base_embed_model,model,tokenizer,nodes,data,flag="original")


def train_eval_pipeline(finetune_class:str="adapter", # adapter,sentence_transformers
                  adapter_class: str="Linear", # None for linear, "TwoLayerNN"
                  model_name:str=EMBED_MODEL_PATH,
                  model_output_path:str="model_output/bge_large_cn_adapter_linear",
                  use_instruction:bool=False):
    
    li_train_finetune(
                     model_name=model_name,
                     model_output_path=model_output_path,
                     engine_class=finetune_class,
                     adapter_model=adapter_class,
                     use_instruction=use_instruction
                     )
    
    li_eval_finetune(
                     finetune_class=finetune_class,
                     adapter_class=adapter_class,
                     model_name=model_name,
                     model_output_path=model_output_path,
                     use_instruction=use_instruction
                     )
    
if __name__ == "__main__":
    # li_train_finetune()
    li_eval_finetune()
    
