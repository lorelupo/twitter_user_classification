# DADIT
## A Dataset for Demographic Classification of Italian Twitter Users and a Comparison of Prediction Methods.

Classify you twitter users' gender and age based on their profile information and tweets, using either fine-tuned classifiers or generative language models (LM) in the zero-shot and few-shot setting.

## Requirements and Installation

TODO

## Fine-tuned Classifiers

TODO


## Generative Language Models

Gender and age classification can also be performed in the zero-shot and few-shot setting with generative language models from HuggingFace or the OpenAI API.

In this section, we describe a use case: zero-shot age classification with Flan-T5 from Hugging Face, using both the users' bio and tweets as features.

1. Define a prompt in a `.txt` file. E.g., [/instructions/age_classification/hf_4groups_bio_tweets.txt](/instructions/age_classification/hf_4groups_bio_tweets.txt):
    ```
    Classify the age of the author of the following texts across 4 age groups: "0-19", "20-29", "30-39", "40-100". Texts:
    """
    ```

2. Define a task in a `.json` file, describing the labels provided to the LM in the prompt as keys, their corresponding form in the test set (if any) as values (otherwise values can be the same as keys). E.g., [tasks/age_classification/bio_tweets.json](tasks/age_classification/bio_tweets.json):

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
    The data-reading function needs to be defined in [task_manager.py](task_manager.py) as a static method. See the methods that are already defined as an example.
3. Run...











