""" 用于基于非对称标注数据的adapter微调，基于llama-index实现"""

from llama_index.finetuning import (SentenceTransformersFinetuneEngine,
                                    EmbeddingAdapterFinetuneEngine)
from llama_index.embeddings import resolve_embed_model
from llama_index.embeddings.adapter_utils import (TwoLayerNN,
                                                  BaseAdapter)
from data_processor import union_data_process, adapter_data_process


def st_finetuning(model_name:str="local:BAAI/bge-large-cn",
                  model_output_path:str="bge_large_cn_adapter",
                  adapter_model:str="TwoLayerNN",
                  use_instruction:bool=False):
    adapter_model = TwoLayerNN(
    384,  # input dimension
    1024,  # hidden dimension
    384,  # output dimension
    bias=True,
    add_residual=True,
        )
    data = union_data_process(process_func=adapter_data_process,
                              combine_flag=False,
                              instruction_flag=use_instruction,
                              )
    base_embed_model = resolve_embed_model("local:BAAI/bge-small-en")

    finetune_engine = EmbeddingAdapterFinetuneEngine(
        data,
        base_embed_model,
        model_output_path=model_output_path,
        # model_checkpoint_path="model5_ck",
        adapter_model=adapter_model,
        epochs=25,
        verbose=True,
    )
    finetune_engine.finetune()
    embed_model_2layer = finetune_engine.get_finetuned_model(
    adapter_cls=TwoLayerNN
    )

    #     embed_model_2layer = AdapterEmbeddingModel(
    #     base_embed_model,
    #     "model5_output_test",
    #     TwoLayerNN,
    # )
    
    

    
