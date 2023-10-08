#!/bin/bash

# model_name=text-davinci-003
# model_name=gpt-4
# max_len_model=4097
# max_len_model=4097
model_name=google/flan-t5-xxl
max_len_model=512

for instruction in  twitter_gender_classification/bio_hf.txt
do
python main.py \
    --data_file data/user_classification/data_for_models_test.pkl \
    --instruction instructions/$instruction \
    --task_file tasks/twitter_gender_classification/bio.json \
    --prompt_suffix "\\n\"\"\"\\nGender:" \
    --model_name $model_name \
    --max_len_model $max_len_model \
    --output_dir tmp \
    --cache_dir /scratch/mentalism/cache/ \
    --evaluation_only False
done