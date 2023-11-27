nohup python data_processor.py --process_func supervised_data_process \
                                --combine_flag \
                                --process_args '{" file_name":"supervised_finetune_data.jsonl"}' \
                                --return_amount train >data_process.log 2>&1 &
