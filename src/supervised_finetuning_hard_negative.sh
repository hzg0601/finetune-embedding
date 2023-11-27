python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path BAAI/bge-base-en-v1.5 \ 
--input_file supervised_finetune_data.jsonl \
--output_file supervised_finetune_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--use_gpu_for_searching