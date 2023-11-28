script_dir=$(readlink -f "$0")
current_dir=$(dirname $(dirname $script_dir))
echo currrent dir is $current_dir
echo $current_dir/data/supervised_finetune_data.jsonl 

python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path /home/star/models/bge-large-zh-v1.5 \
--output_file $current_dir/data/supervised_finetune_data_minedHN.jsonl \
--input_file $current_dir/data/supervised_finetune_data.jsonl \
--range_for_sampling 2-200 \
--use_gpu_for_searching