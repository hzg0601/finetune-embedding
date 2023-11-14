""" 用于基于非对称标注数据的adapter微调，基于llama-index实现"""
import os
from utils import logger
from typing import Union
from llama_index.finetuning import (SentenceTransformersFinetuneEngine,
                                    EmbeddingAdapterFinetuneEngine)
from llama_index.embeddings import (resolve_embed_model,
                                    AdapterEmbeddingModel,
                                    HuggingFaceEmbedding,)
from transformers import AutoModel, AutoModelForCausalLM,AutoTokenizer
from llama_index.llms import HuggingFaceLLM
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings.adapter_utils import TwoLayerNN
from llama_index.evaluation import RetrieverEvaluator
from data_processor import union_data_process, adapter_data_process
import pandas as pd


os.environ["TRANSFORMERS_OFFLINE"] = '1'
LLM_MODEL_PATH = "/alidata/models/qwen/Qwen-14B-Chat-Int4/"

def li_train_finetune(model_name:str="local:/home/m3e-base",
                  model_output_path:str="model_output/bge_large_cn_adapter_linear", # 保存最终的checkpoint的路径
                  model_checkpoint_path:str=None, # 保存中间checkpoint文件的路径
                  engine_class:str="adapter",
                  adapter_model:str=None,
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
    
async def li_eval_finetune(finetune_class:str="adapter",
                  adapter_class: str="TwoLayerNN",
                  model_name:str="local:/home/m3e-base",
                  model_output_path:str="model_output/bge_large_cn_adapter_linear",
                  use_instruction:bool=False
                  ):
    data = union_data_process(process_func=adapter_data_process,
                              combine_flag=True,
                              model_output_path=model_output_path,
                              instruction_flag=use_instruction,
                              return_amount="eval"
                              )
    
    if finetune_class == "adapter":
        base_embed_model = resolve_embed_model(model_name)
        adapter_class = eval(adapter_class) if adapter_class is not None else None
        embed_model = AdapterEmbeddingModel(
            base_embed_model=base_embed_model,
            adapter_path=model_output_path,
            adapter_cls=adapter_class
        )
    else:
        embed_model = HuggingFaceEmbedding(model_name=model_output_path)
    nodes = data.corpus
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH,trust_remote_code=True)
    llm = HuggingFaceLLM(model=model,tokenizer=tokenizer)
    service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
    vector_index = VectorStoreIndex(nodes, service_context=service_context)
    retriever = vector_index.as_retriever(similarity_top_k=2)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
    )
    eval_results = await retriever_evaluator.aevaluate_dataset(data)
    display_results("top-2 eval", eval_results)


if __name__ == "__main__":
    li_train_finetune()
    li_eval_finetune()
    
