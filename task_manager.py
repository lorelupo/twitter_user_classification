import json
import re
import pandas as pd

class TaskManager:
    """
    Utility class for defining labels and reading data tables (.csv, .tsv, ...) and returning the input texts and gold labels
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
            "twitter_features_extra_nogold": TaskManager.twitter_features_extra_nogold,
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
        elif path_data.endswith('.parquet'):
            df = pd.read_parquet(path_data)
        else:
            raise ValueError('The file format is not supported. Please use .pkl or .parquet files')
        
        # Set the index to the user_id
        df.set_index('user_id', inplace=True)
        # check if there are any missing values (shouldn't be the case)
        if df.isnull().values.any():
            raise ValueError('The dataframe contains missing values')
        # Read each bio and tweets concatenation, splitting them by \n and
        # joining by '. ' if sentences don't already end with a dot, else join by ' '
        if include_bio:
            df.masked_bio = df.masked_bio.apply(lambda x: [text + '.' if not (text.endswith('.') or text.endswith('!') or text.endswith('?') or text.endswith(';')) else text for text in x.split('\n')]).apply(lambda x: ' '.join(x)).apply(lambda x: re.sub('\r', '', x)).tolist()
        if include_tweets:
            df.long_text = df.long_text.apply(lambda x: [text + '.' if not (text.endswith('.') or text.endswith('!') or text.endswith('?') or text.endswith(';')) else text for text in x.split('\n')]).apply(lambda x: ' '.join(x)).apply(lambda x: re.sub('\r', '', x)).tolist()
        if include_bio and include_tweets:
            # Join each tweet and bio by 'Bio: ' and 'Tweets: '
            input_texts = df.apply(lambda x: 'Bio: ' + x.masked_bio + '\nTweets: ' + x.long_text, axis=1)
        elif include_bio:
            input_texts = df.apply(lambda x: 'Bio: ' + x.masked_bio, axis=1)
        elif include_tweets:
            input_texts = df.apply(lambda x: 'Tweets: ' + x.long_text, axis=1)

        # Read the gold labels
        if label_name == 'is_male':
            gold_labels = df[[label_name]]
            gold_labels[label_name] = gold_labels[label_name].apply(lambda x: 'male' if x==True else 'female')
        if label_name == 'age':
            gold_labels = df[[label_name]]
            gold_labels[label_name] = gold_labels[label_name].astype('int')
        if label_name == 'age_interval':
            # define age classes
            age_intervals = [0, 19, 30, 40, 100]
            age_labels = [0, 1, 2, 3]
            # Discretize the 'age' column into four classes
            gold_labels = pd.cut(df['age'], bins=age_intervals, labels=age_labels, right=False).astype('str')
        
        return input_texts, gold_labels

    @staticmethod
    def twitter_features_extra_nogold(
        path_data: str,
        ):

        # Read the pickle dataframe
        if path_data.endswith('.pkl'):
            df = pd.read_pickle(path_data)
        elif path_data.endswith('.parquet'):
            df = pd.read_parquet(path_data)
        else:
            raise ValueError('The file format is not supported. Please use .pkl or .parquet files')

        # Set the index to the user_id
        df.set_index('user_id', inplace=True)
        # create input text
        # Separating text and numbers with a space
        df['username_sep'] = df['username'].str.replace(r'([a-zA-Z])(\d)', r'\1 \2').\
                            str.replace(r'(\d)([a-zA-Z])', r'\1 \2')
        # concat info
        df['input_texts']  = 'NAME:' + ' "' + df['full_name'] + '". ' +\
                            'USERNAME:' + ' "'+  df['username_sep'] + '". ' + \
                            'JOINED:' + ' "' + df['join_year'].astype(str) + '". ' +\
                            'TWEETS:' + ' "' + df['tweets'].astype(str) + '". ' + \
                            'FOLLOWING:' + ' "' + df['following'].astype(str) + '". ' +\
                            'FOLLOWERS:' + ' "' + df['followers'].astype(str) + '". ' + \
                            'BIO:' + ' "' + df['bio'] + '". ' + \
                            'TEXT:' + ' "' + df['long_text'] + '".'

        # check if there are any missing values in input texts (shouldn't be the case)
        if df.input_texts.isnull().values.any():
            raise ValueError('The dataframe contains missing input_texts')

        return df['input_texts'], None

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
