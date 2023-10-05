import json
import pandas as pd

class TaskManager:
    """
    Utility class for reading data tables (.csv, .tsv, ...) and returning the input texts and gold labels
    for an LLM classifier.
    """

    def __init__(self, task_file, logdir=None):

        # define dictionary of data-reading functions
        self.data_reading_functions = {
            "pappa": self.read_data_pappa,
            "twitter_bio": self.twitter_bio,
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
    def read_data_pappa(path_data):
        # Read the csv/xlsx data file to table
        if path_data.endswith('.csv'):
            df = pd.read_csv(path_data, sep=';').fillna('NA')
        elif path_data.endswith('.xlsx'):
            df = pd.read_excel(path_data).fillna('NA')
        # Read texts
        input_texts = [text[:-1] + '.' if not text.endswith('.') else text for text in df['text_clean'].tolist()]
        # Read gold labels if any
        if 'elin' in df.columns:
            gold_labels = df[['elin', 'lena', 'oscar', 'agg']]
        else:
            gold_labels = None
        return input_texts, gold_labels

    @staticmethod
    def twitter_bio(path_data):
        raise NotImplementedError