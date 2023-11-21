# DADIT
## A Dataset for Demographic Classification of Italian Twitter Users and a Comparison of Prediction Methods.

Classify you twitter users' gender and age based on their profile information and tweets, using either fine-tuned classifiers or generative language models (LM) in the zero-shot and few-shot setting.

## Install

```
git clone git@github.com:lorelupo/twitter_user_classification.git
cd twitter_user_classification
pip install -r ./requirements.txt
```

## Gender and Age Classification

Classification of users' gender and age attributes can be done with either [one of our fine-tuned classifiers stored on HuggingFace](https://huggingface.co/lorelupo):

```
python classification_discriminative.py \
    --data_file data/user_classification/data_for_models_test.pkl \
    --task_file tasks/gender_classification/bio_tweets_int.json \
    --model_name lorelupo/twitter-xlm-gender-prediction-italian \
    --max_len_model 512 \
    --output_dir tmp \
    --cache_dir ~/.cache/huggingface/hub/
```

Or with generative LMs hosted on HuggingFace, in a zero/few-shot setting:

```
python classification_generative.py \
    --data_file data/user_classification/data_for_models_test.pkl \
    --task_file tasks/gender_classification/bio_tweets.json \
    --instruction instructions/gender_classification/bio_tweets_hf.txt \
    --prompt_suffix \\n\"\"\"\\nGender: \
    --model_name google/flan-t5-xxl \
    --max_len_model 512 \
    --output_dir tmp \
    --cache_dir /scratch/mentalism/cache/
```

Or with generative LMs by OpenAI, in a zero/few-shot setting:

```
python classification_generative.py \
    --data_file data/user_classification/data_for_models_test.pkl \
    --task_file tasks/gender_classification/bio_tweets.json \
    --instruction instructions/gender_classification/gpt_fewshot_bio_tweets_it.txt \
    --prompt_suffix \\nGender: \
    --model_name gpt-3.5-turbo \
    --max_len_model 2048 \
    --output_dir tmp
```

The  available tasks are:
    
- `gender_classification/`:
    - `bio`: only the users' bio
    - `bio_tweeets`: both the users' bio and tweets
    - `bio_tweeets_int`: : both the users' bio and tweets, when the labels output by the classifier are the integer number of the class (e.g., 0/1 instead of "male"/"female")
- `age_classification`, classifying users' age in 4 groups given the following information as features: 
    - `bio`: only the users' bio
    - `bio_tweeets`: both the users' bio and tweets
    - `bio_tweeets_int`: both the users' bio and tweets, when the labels output by the classifier are the integer number of the class (e.g., 0/1/2/3 instead of "0-19"/"20-29"/"30-39"/"40-100")

Check the folder [instructions](instructions) to see available instructions for generative LMs and add new ones.

## Defining a Task

It is possible to define new classification tasks by creating a new `.json` file describing the dictionary of labels and the data-reading function. See the age classification task defined in [tasks/age_classification/bio_tweets.json](tasks/age_classification/bio_tweets.json) as an example:

```json
{
    "labels": {
        "0-19": "0",
        "20-29": "1",
        "30-39": "2",
        "40-100": "3"
        },
    "read_function": "twitter_features_age_interval_bio_tweets"
}
```

In the labels dictionary, the keys are the labels in the format output by the classifier, while the values are the labels as represented in your data. In this case, a classifier outputs an integer referring to the age group of the Twitter user.

The data-reading function needs to be defined in the [task_manager.py](task_manager.py) as a static method. For instance, see the definition of [twitter_features_age_interval_bio_tweets](twitter_features_age_interval_bio_tweets.py?plain=1#L105), a utility function that reads [data/user_classification/data_for_models_test.pkl](data/user_classification/data_for_models_test.pkl) and creates a string containing both users' bio and tweets as a feature for the classifier.

## Citation

TODO