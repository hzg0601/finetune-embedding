# 注意要项目根目录上运行
torchrun --nproc_per_node 1 \
-m FlagEmbedding.baai_general_embedding.retromae_pretrain.run \
--output_dir ./model_output/unsupervised_train/ \
--model_name_or_path BAAI/bge-large-zh-v1.5 \
--train_data supervised_finetune_data_minedHN.jsonl \
--learning_rate 2e-5 \
--num_train_epochs 2 \
--per_device_train_batch_size 4 \
--dataloader_drop_last True \
--max_seq_length 512 \
--logging_steps 10 \
--dataloader_num_workers 12
