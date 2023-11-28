script_dir=$(readlink -f "$0")
current_dir=$(dirname $(dirname $script_dir))
echo currrent dir is $current_dir

torchrun --nproc_per_node 1 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir $current_dir/model_output/supervised_train/ \
--model_name_or_path $HOME/models/bge-large-zh-v1.5 \
--train_data $current_dir/data/supervised_finetune_data_minedHN.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval "" 