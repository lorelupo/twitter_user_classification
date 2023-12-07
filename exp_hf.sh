#!/bin/bash

# model_name=text-davinci-003
# model_name=gpt-4
# max_len_model=4097
# max_len_model=4097
model_name=google/flan-t5-xxl
max_len_model=512


CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_file data/user_classification/data_for_models_test.pkl \
    --instruction instructions/gender_classification/bio_hf.txt \
    --task_file tasks/gender_classification/bio.json \
    --prompt_suffix \\n\"\"\"\\nGender: \
    --model_name $model_name \
    --max_len_model $max_len_model \
    --output_dir tmp \
    --cache_dir /data/mentalism/cache/

CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_file data/user_classification/data_for_models_test.pkl \
    --instruction instructions/gender_classification/bio_tweets_hf.txt \
    --task_file tasks/gender_classification/bio_tweets.json \
    --prompt_suffix \\n\"\"\"\\nGender: \
    --model_name $model_name \
    --max_len_model $max_len_model \
    --output_dir tmp \
    --cache_dir /data/mentalism/cache/

CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_file data/user_classification/data_for_models_test.pkl \
    --instruction instructions/age_classification/bio_hf.txt \
    --task_file tasks/age_classification/bio.json \
    --prompt_suffix \\n\"\"\"\\nAge\ group: \
    --model_name $model_name \
    --max_len_model $max_len_model \
    --output_dir tmp \
    --cache_dir /data/mentalism/cache/

CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_file data/user_classification/data_for_models_test.pkl \
    --instruction instructions/age_classification/bio_tweets_hf.txt \
    --task_file tasks/age_classification/bio_tweets.json \
    --prompt_suffix \\n\"\"\"\\nAge\ group: \
    --model_name $model_name \
    --max_len_model $max_len_model \
    --output_dir tmp \
    --cache_dir /data/mentalism/cache/


model_name=lorelupo/twitter-xlm-large-user-age-5g-it-extra
max_len_model=512
CUDA_VISIBLE_DEVICES=2 python classification_discriminative.py \
    --data_file /data/mentalism/data/user_classification/user_regioncoded_features_sample.pkl \
    --task_file tasks/age_classification/5g_extra_nogold.json \
    --model_name $model_name \
    --max_len_model $max_len_model \
    --output_dir tmp \
    --batch_size 32 \
    --cache_dir /data/mentalism/cache/huggingface/

model_name=lorelupo/twitter-xlm-large-user-gender-it-extra
max_len_model=512
CUDA_VISIBLE_DEVICES=3 python classification_discriminative.py \
    --data_file /data/mentalism/data/user_classification/user_regioncoded_features_sample.pkl \
    --task_file tasks/gender_classification/extra_int_nogold.json \
    --model_name $model_name \
    --max_len_model $max_len_model \
    --output_dir tmp \
    --batch_size 32 \
    --cache_dir /data/mentalism/cache/huggingface/