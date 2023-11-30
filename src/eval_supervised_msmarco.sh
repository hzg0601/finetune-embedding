# huggingface-cli download namespace-Pt/msmarco-corpus --resume-download --repo-type datasets  && \
# huggingface-cli download namespace-Pt/msmarco-corpus --resume-download --repo-type datasets  && \
script_dir=$(readlink -f "$0")
current_dir=$(dirname $(dirname $script_dir))
echo currrent dir is $current_dir

# eval finetuned model on public data 
python -m src/eval_msmarco \
--encoder $current_dir/model_output/supervised_train \
--fp16 \
--add_instruction \
--k 100 >eval_msmarco.log 2>&1 

# eval finetuned model on private data 
python -m src/eval_msmarco \
--encoder $current_dir/model_output/supervised_train \
--fp16 \
--add_instruction \
-- private_data_path $current_dir/data/supervised_finetune_data.jsonl
--k 100 >eval_private.log 2>&1

# eval finetuned model on private data, but add some public data as candidate corpus 
python -m src/eval_msmarco \
--encoder $current_dir/model_output/supervised_train \
--fp16 \
--add_instruction \
-- private_data_path $current_dir/data/supervised_finetune_data.jsonl
--k 100 >eval_private.log 2>&1 

# -------------------------------------------------------
# eval base model on public data 
python -m src/eval_msmarco \
--encoder /home/star/models/bge-large-zh-v1.5 \
--fp16 \
--add_instruction \
--k 100 >eval_msmarco.log 2>&1 

# eval base model on private data 
python -m src/eval_msmarco \
--encoder /home/star/models/bge-large-zh-v1.5 \
--fp16 \
--add_instruction \
-- private_data_path $current_dir/data/supervised_finetune_data.jsonl
--k 100 >eval_private.log 2>&1

# eval base model on private data, but add some public data as candidate corpus 
python -m src/eval_msmarco \
--encoder /home/star/models/bge-large-zh-v1.5 \
--fp16 \
--add_instruction \
--private_data_path $current_dir/data/supervised_finetune_data.jsonl
--add_extra_corpus True
--k 100 >eval_private.log 2>&1 