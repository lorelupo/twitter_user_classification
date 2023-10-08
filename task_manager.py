import json
import re
import pandas as pd

class TaskManager:
    """
    Utility class for reading data tables (.csv, .tsv, ...) and returning the input texts and gold labels
    for an LLM classifier.
    """

    def __init__(self, task_file, logdir=None):

        # define dictionary of data-reading functions
        self.data_reading_functions = {
            "twitter_features_gender_bio": TaskManager.twitter_features_gender_bio,
            "twitter_features_gender_bio_tweets": TaskManager.twitter_features_gender_bio_tweets,
            "twitter_features_age_bio": TaskManager.twitter_features_age_bio,
            "twitter_features_age_bio_tweets": TaskManager.twitter_features_age_bio_tweets,
            "twitter_features_age_interval_bio": TaskManager.twitter_features_age_interval_bio,
            "twitter_features_age_interval_bio_tweets": TaskManager.twitter_features_age_interval_bio_tweets,
        }

        # read task specs from json task_file
        self.task = json.load(open(task_file, 'r'))
        # setup labels
        self.labels = self.task['labels']
        self.label_dims = self.task['label_dims'] if 'label_dims' in self.task else 1
        self.default_label = list(self.labels.keys())[0] if self.label_dims == 1 else list(self.labels['dim1'].keys())[0]
        # setup data reading function
        if self.task['read_function'] in self.data_reading_functions:
            self.read_data = self.data_reading_functions[self.task['read_function']]
        else:
            raise ValueError(
                f"Data-reading function '{self.task['read_function']}' not supported."
                f"Supported functions: {self.data_reading_functions.keys()}"
                )

    @staticmethod
    def twitter_features(
        path_data: str,
        include_bio=True,
        include_tweets=True,
        label_name='is_male',
        ):
        # Read the pickle dataframe
        if path_data.endswith('.pkl'):
            df = pd.read_pickle(path_data)
        else:
            raise NotImplementedError
        # check if there are any missing values (shouldn't be the case)
        if df.isnull().values.any():
            raise ValueError('The dataframe contains missing values')
        # Read each bio and tweets concatenation, splitting them by \n and
        # joining by '. ' if sentences don't already end with a dot, else join by ' '
        if include_bio:
            bios = df.masked_bio.apply(lambda x: [text + '.' if not (text.endswith('.') or text.endswith('!') or text.endswith('?') or text.endswith(';')) else text for text in x.split('\n')]).apply(lambda x: ' '.join(x)).apply(lambda x: re.sub('\r', '', x)).tolist()
        if include_tweets:
            tweets = df.long_text.apply(lambda x: [text + '.' if not (text.endswith('.') or text.endswith('!') or text.endswith('?') or text.endswith(';')) else text for text in x.split('\n')]).apply(lambda x: ' '.join(x)).apply(lambda x: re.sub('\r', '', x)).tolist()
        if include_bio and include_tweets:
            # Join each tweet and bio by 'Bio: ' and 'Tweets: '
            input_texts = ['Bio: ' + bio + '\nTweets: ' + tweet for bio, tweet in zip(bios, tweets)]
        elif include_bio:
            input_texts = ['Bio: ' + bio for bio in bios]
        elif include_tweets:
            input_texts = ['Tweets: ' + tweet for tweet in tweets]

        # Read the gold labels
        if label_name == 'is_male':
            gold_labels = df[label_name].tolist()
            gold_labels = ['male' if label == True else 'female' for label in gold_labels]
        if label_name == 'age':
            gold_labels = df[label_name].astype(int).tolist()
        if label_name == 'age_interval':
            # define age classes
            age_intervals = [0, 19, 30, 40, 100]
            age_labels = [0, 1, 2, 3]
            # Discretize the 'age' column into four classes
            gold_labels = pd.cut(df['age'], bins=age_intervals, labels=age_labels, right=False).tolist()
        
        return input_texts, gold_labels

    @staticmethod
    def twitter_features_gender_bio(path):
        return TaskManager.twitter_features(path, include_bio=True, include_tweets=False, label_name='is_male')

    @staticmethod
    def twitter_features_gender_bio_tweets(path):
        return TaskManager.twitter_features(path, include_bio=True, include_tweets=True, label_name='is_male')

    @staticmethod
    def twitter_features_age_bio(path):
        return TaskManager.twitter_features(path, include_bio=True, include_tweets=False, label_name='age')

    @staticmethod
    def twitter_features_age_bio_tweets(path):
        return TaskManager.twitter_features(path, include_bio=True, include_tweets=True, label_name='age')

    @staticmethod
    def twitter_features_age_interval_bio(path):
        return TaskManager.twitter_features(path, include_bio=True, include_tweets=False, label_name='age_interval')

    @staticmethod
    def twitter_features_age_interval_bio_tweets(path):
        return TaskManager.twitter_features(path, include_bio=True, include_tweets=True, label_name='age_interval')
