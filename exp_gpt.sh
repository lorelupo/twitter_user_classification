#!/bin/bash

# model_name=text-davinci-003
# model_name=gpt-4
# max_len_model=4097
model_name=gpt-3.5-turbo
max_len_model=512

# python main.py \
#     --data_file data/user_classification/data_for_models_test.pkl \
#     --instruction instructions/gender_classification/bio_gpt.txt \
#     --task_file tasks/gender_classification/bio.json \
#     --prompt_suffix \\n\"\"\"\\nGender: \
#     --model_name $model_name \
#     --max_len_model $max_len_model \
#     --output_dir tmp \
#     --sleep_after_step 0

python main.py \
    --data_file data/user_classification/data_for_models_test.pkl \
    --instruction instructions/gender_classification/bio_tweets_gpt.txt \
    --task_file tasks/gender_classification/bio_tweets.json \
    --prompt_suffix \\n\"\"\"\\nGender: \
    --model_name $model_name \
    --max_len_model $max_len_model \
    --output_dir tmp \
    --sleep_after_step 0