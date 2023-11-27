#!/bin/bash

# model_name=text-davinci-003
# model_name=gpt-4
# max_len_model=4097
# max_len_model=4097
model_name=google/flan-t5-xxl
max_len_model=512

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --data_file data/user_classification/data_for_models_test.pkl \
#     --instruction instructions/gender_classification/bio_hf.txt \
#     --task_file tasks/gender_classification/bio.json \
#     --prompt_suffix \\n\"\"\"\\nGender: \
#     --model_name $model_name \
#     --max_len_model $max_len_model \
#     --output_dir tmp \
#     --cache_dir /data/mentalism/cache/ \
#     --evaluation_only False

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --data_file data/user_classification/data_for_models_dutch_data.pkl \
#     --instruction instructions/gender_classification/bio_tweets_hf.txt \
#     --task_file tasks/gender_classification/bio_tweets.json \
#     --prompt_suffix \\n\"\"\"\\nGender: \
#     --model_name $model_name \
#     --max_len_model $max_len_model \
#     --output_dir tmp_nl \
#     --cache_dir /data/mentalism/cache/ \
#     --evaluation_only False


CUDA_VISIBLE_DEVICES=1 python main.py \
    --data_file data/user_classification/data_for_models_dutch_data.pkl \
    --instruction instructions/age_classification/bio_tweets_hf.txt \
    --task_file tasks/age_classification/bio_tweets.json \
    --prompt_suffix \\n\"\"\"\\nAge\ group: \
    --model_name $model_name \
    --max_len_model $max_len_model \
    --output_dir tmp_nl\
    --cache_dir /data/mentalism/cache/