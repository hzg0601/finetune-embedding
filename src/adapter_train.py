""" 用于基于非对称标注数据的adapter微调，基于llama-index实现"""
from utils import logger
import os
from llama_index.finetuning import (SentenceTransformersFinetuneEngine,
                                    EmbeddingAdapterFinetuneEngine)
from llama_index.embeddings import resolve_embed_model
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.embeddings.adapter_utils import (TwoLayerNN,
                                                  BaseAdapter)
from data_processor import union_data_process, adapter_data_process

os.environ["TRANSFORMERS_OFFLINE"] = '1'

def adapter_finetune(model_name:str="local:/home/m3e-base",
                  model_output_path:str="bge_large_cn_adapter",
                  model_checkpoint_path:str="twolayer_bge_large_cn",
                  adapter_model:str="TwoLayerNN",
                  use_instruction:bool=False
                  ):
    logger.info("llama-index adapter fine-tuning start...")

    data = union_data_process(process_func=adapter_data_process,
                              combine_flag=False,
                              instruction_flag=use_instruction,
                              )
    logger.info("data procession done..")
    # resolve_embed_model(model_name),model_name必须是local: repo_id的格式
    base_embed_model = resolve_embed_model(model_name)
    if adapter_model is not None:
        adapter_model = eval(adapter_model)
        adapter_model_ins = adapter_model(
        in_features=384,  # input dimension
        hidden_features=1024,  # hidden dimension
        out_features=384,  # output dimension
        bias=True,
        add_residual=True,
            )
    else:
        adapter_model_ins = None    

    finetune_engine = EmbeddingAdapterFinetuneEngine(
        data,
        base_embed_model,
        model_output_path=model_output_path,
        model_checkpoint_path=model_checkpoint_path,
        adapter_model=adapter_model_ins,
        epochs=25,
        verbose=True,
    )
    logger.info("fine-tuning start...")
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model(
    adapter_cls=TwoLayerNN
    )
    logger.info("llama-index adapter fine-tuning done.")
    return embed_model
    #     embed_model_2layer = AdapterEmbeddingModel(
    #     base_embed_model,
    #     "model5_output_test",
    #     TwoLayerNN,
    # )
    
    
def sf_finetune():
    pass

if __name__ == "__main__":
    adapter_finetune()
    
