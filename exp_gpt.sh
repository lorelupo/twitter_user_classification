#!/bin/bash

# model_name=text-davinci-003
# model_name=gpt-4
# max_len_model=4097
# max_len_model=4097
model_name=gpt-3.5-turbo
max_len_model=4097

for instruction in  twitter_gender_classification/bio_gpt.txt
do
python main.py \
    --data_file data/user_classification/data_for_models_test.pkl \
    --instruction instructions/$instruction \
    --task_file tasks/twitter_gender_classification/bio.json \
    --prompt_suffix "\\n\"\"\"\\nGender:" \
    --model_name $model_name \
    --max_len_model $max_len_model \
    --output_dir tmp \
    --sleep_after_step 0
done