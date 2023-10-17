import openai
import backoff
from dotenv import load_dotenv
import os
import re
import time
import pandas as pd
import collections 
import torch
from utils import setup_logging
from logging import getLogger, StreamHandler
logger = getLogger(__name__)
logger_backoff = getLogger('backoff').addHandler(StreamHandler())

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

class LMClassifier:
    def __init__(
            self,
            labels_dict,
            label_dims,
            default_label,
            instruction,
            prompt_suffix,
            model_name,
            max_len_model,
            output_dir=None,
            log_to_file=True
            ):

        setup_logging(os.path.basename(__file__).split('.')[0], logger, output_dir if log_to_file else None)

        self.labels_dict = labels_dict
        # check the dimensionality of the labels:
        # dimensionality greater than 1 means dealing with
        # multiple classification tasks at a time
        self.label_dims = label_dims
        assert self.label_dims > 0, "Labels dimensions must be greater than 0."
        self.default_label = default_label
        
        # Define the instruction and ending ending string for prompt formulation
        # If instruction is a path to a file, read the file, else use the instruction as is
        self.instruction = open(instruction, 'r').read() if os.path.isfile(instruction) else instruction
        self.prompt_suffix = prompt_suffix.replace('\\n', '\n')

        self.max_len_model = max_len_model
        self.model_name = model_name

    def generate_predictions(self):
        raise NotImplementedError
    

    def range_robust_get_label(self, prediction, bounds):
        # more robust get label function that manages numbers in the returned text and assigns them to the correct range in case of number ranges
        # extract all two digit numbers or 0 from the prediction
        numbers = [int(n) for n in re.findall('\d{2}|[0]',prediction)]
        if len(numbers)==0:
            return self.labels_dict.get(self.default_label)
        if len(numbers)>0:
            if (numbers[-1]>bounds[-1][-1]) or (numbers[0]<bounds[0][0]):
                return self.labels_dict.get(self.default_label)
            elif len(numbers)==1:
                #check which list in bounds the number belongs to
                for i, bound in enumerate(bounds):
                    if numbers[0] in bound:
                        return self.labels_dict.get(list(self.labels_dict.keys())[i])
            elif len(numbers)>1:
                #just use the first 2 numbers
                # check the overlap of the range between numbers with bounds
                overlaps = [len(set(range(numbers[0],numbers[1])).intersection(set(bound))) for bound in bounds]
                return self.labels_dict.get(list(self.labels_dict.keys())[overlaps.index(max(overlaps))])


    def retrieve_predicted_labels(self, predictions, prompts, only_dim=None):

        # convert the predictions to lowercase
        predictions =  list(map(str.lower,predictions))

        # retrieve the labels that are contained in the predictions
        predicted_labels = []
        if self.label_dims == 1:
            # retrieve a single label for each prediction since a single classification task is performed at a time
            logger.info("Retrieving predictions...")
            for prediction in predictions:
                labels_in_prediction = [self.labels_dict.get(label) for label in self.labels_dict.keys() if label in prediction.split()]
                if len(labels_in_prediction) > 0:
                    predicted_labels.append(labels_in_prediction[0])
                else:
                    # first check if there is a range in all the labels
                    bounds = [[int(n) for n in key.split('-') if n.isnumeric()] for key in self.labels_dict.keys()]
                    if all(bounds): #if all labels have a number range
                        bounds = [list(range(b[0],b[1]+1)) for b in bounds]
                        predicted_labels.append(self.range_robust_get_label(prediction,bounds))
                    else:
                        predicted_labels.append(self.labels_dict.get(self.default_label))
            # Count the number of predictions of each type and print the result
            logger.info(collections.Counter(predicted_labels))
        else:
            # retrieve multiple labels for each prediction since multiple classification tasks are performed at a time
            logger.info(f"Retrieving predictions for {self.label_dims} dimensions...")
            for prediction in predictions:
                labels_in_prediction = []
                for dim in self.labels_dict.keys():
                    dim_label = []
                    for label in self.labels_dict[dim].keys():
                        if label in prediction:
                            dim_label.append(self.labels_dict[dim].get(label))   
                    dim_label = dim_label[0] if len(dim_label) > 0 else self.labels_dict[dim].get(self.default_label)
                    labels_in_prediction.append(dim_label)                                            
                predicted_labels.append(labels_in_prediction)
            # Count the number of predictions of each type and print the result
            logger.info(collections.Counter([",".join(labels) for labels in predicted_labels]))
        
        # Add the data to a DataFrame
        if self.label_dims == 1:
            df = pd.DataFrame({'prompt': prompts, 'prediction': predicted_labels})
        elif self.label_dims > 1:
            if only_dim is not None:
                # retrieve only the predictions for a specific dimension
                logger.info(f"Retrieved predictions for dimension {only_dim}")
                df = pd.DataFrame({'prompt': prompts, 'prediction': pd.DataFrame(predicted_labels).to_numpy()[:,only_dim]})
            else:
                logger.info("Retrieved predictions for all dimensions")
                df = pd.DataFrame(predicted_labels).fillna(self.default_label)
                # rename columns to prediction_n
                df.columns = [f"prediction_dim{i}" for i in range(1, len(df.columns)+1)]
                # add prompts to df
                df['prompt'] = prompts

        return df

    def evaluate_predictions(self, df, gold_labels, aggregated_gold_name='agg'):
        """
        Evaluate the predictions of a model, stored in the df, against gold labels.
        The df contains the following columns:
        - prompt: the prompt used to generate the prediction
        - prediction: the prediction. If the model performs multiple classification tasks at a time,
          the df contains multiple columns named prediction_dim1, prediction_dim2, etc. 
        """

        # Add the gold labels to df
        if isinstance(gold_labels, pd.DataFrame):
            for col in gold_labels.columns:
                df['gold_' + col] = gold_labels[col]    
        elif isinstance(gold_labels, list):
            df['gold'] = gold_labels
        else:
            raise ValueError('The gold labels must be either a list or a DataFrame.')
        
        logger.info("Evaluating predictions...")
        logger.info(f"\n{df.head()}\n")
        
        # define gold_labels method variable
        gold_labels = df.filter(regex='^gold', axis=1)

        # retrieve the name of each gold annotation
        gold_names = [col.split('gold_')[-1] for col in gold_labels.columns]

        # define tables where to store results
        df_kappa = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)
        df_accuracy = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)
        df_f1 = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)

        for i, col in enumerate(gold_labels.columns):
            # compare agreement with gold labels
            kappa = cohen_kappa_score(df['prediction'].astype(str), gold_labels[col].astype(str))
            accuracy = accuracy_score(df['prediction'].astype(str), gold_labels[col].astype(str))
            f1 = f1_score(df['prediction'].astype(str), gold_labels[col].astype(str), average='macro')
            # store results
            df_kappa.loc['model', gold_names[i]] = df_kappa.loc[gold_names[i], 'model'] = kappa
            df_accuracy.loc['model', gold_names[i]] = df_accuracy.loc[gold_names[i], 'model'] = accuracy
            df_f1.loc['model', gold_names[i]] = df_f1.loc[gold_names[i], 'model'] = f1

            if len(gold_labels.columns) > 1:
                for j, col2 in enumerate(gold_labels.columns):
                    if i < j:
                        # compare agreement of gold labels with each other
                        kappa = cohen_kappa_score(gold_labels[col].astype(str), gold_labels[col2].astype(str))
                        accuracy = accuracy_score(gold_labels[col].astype(str), gold_labels[col2].astype(str))
                        f1 = f1_score(gold_labels[col].astype(str), gold_labels[col2].astype(str), average='macro')
                        # store results
                        df_kappa.loc[gold_names[i], gold_names[j]] = df_kappa.loc[gold_names[j], gold_names[i]] = kappa
                        df_accuracy.loc[gold_names[i], gold_names[j]] = df_accuracy.loc[gold_names[j], gold_names[i]] = accuracy
                        df_f1.loc[gold_names[i], gold_names[j]] = df_f1.loc[gold_names[j], gold_names[i]] = f1

        # in case of multiple gold annotations, there could be a column
        # containing the aggregated annotation (computed with tools like MACE)
        non_agg_names = [name for name in gold_names if aggregated_gold_name not in name]

        # compute average agreement between gold annotations (except the aggregated one)
        if len(gold_labels.columns) > 1:
            df_kappa['mean_non_agg'] = df_kappa[non_agg_names].mean(axis=1)
            df_accuracy['mean_non_agg'] = df_accuracy[non_agg_names].mean(axis=1) 
            df_f1['mean_non_agg'] = df_f1[non_agg_names].mean(axis=1)
            for name in non_agg_names:
                # correct for humans fully agreeing with themselves
                df_kappa.mean_non_agg[name] = (df_kappa[non_agg_names].loc[name].sum() - 1.0) / (len(non_agg_names) - 1.0)
                df_accuracy.mean_non_agg[name] = (df_accuracy[non_agg_names].loc[name].sum() - 1.0) / (len(non_agg_names) - 1.0)
                df_f1.mean_non_agg[name] = (df_f1[non_agg_names].loc[name].sum() - 1.0) / (len(non_agg_names) - 1.0)
        
        # print info
        logger.info(f"KAPPA:\n{df_kappa.round(4)*100}\n")
        if len(gold_labels.columns) > 1:
            logger.info(f"Annotators' mean kappa: {100*df_kappa.mean_non_agg[:-1].mean():.2f}")
            logger.info(f"Model's mean kappa: {100*df_kappa.model[:-1].mean():.2f}")

        logger.info(f"ACCURACY:\n{df_accuracy.round(4)*100}\n")
        if len(gold_labels.columns) > 1:
            logger.info(f"Annotators' mean accuracy: {100*df_accuracy.mean_non_agg[:-1].mean():.2f}") 
            logger.info(f"Model's mean accuracy: {100*df_accuracy.model[:-1].mean():.2f}")

        logger.info(f"F1:\n{df_f1.round(4)*100}\n")

        if len(gold_labels.columns) > 1:
            logger.info(f"Annotators' mean F1: {100*df_f1.mean_non_agg[:-1].mean():.2f}")
            logger.info(f"Model's mean F1: {100*df_f1.model[:-1].mean():.2f}")

        return df_kappa, df_accuracy, df_f1

class GPTClassifier(LMClassifier):
    def __init__(
            self,
            labels_dict,
            label_dims,
            default_label,
            instruction,
            prompt_suffix,
            model_name,
            max_len_model,
            gpt_system_role="You are a helpful assistant.",
            **kwargs,
            ):
        super().__init__(labels_dict, label_dims, default_label, instruction, prompt_suffix, model_name, max_len_model, **kwargs)  
        
        # set the average number of tokens per word in order to compute the max length of the input text
        self.avg_tokens_per_en_word = 4/3 # according to: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        self.avg_tokens_per_nonen_word = 5 # adjust according to the language of the input text
        self.avg_tokens_per_word_avg = (self.avg_tokens_per_en_word + self.avg_tokens_per_nonen_word) / 2

        # if prompt is longer then max_len_model, we will remove words from the imput text
        # differently from HF models, where we have access to the tokenizer, here we work on full words
        len_instruction = len(self.instruction.split())
        len_output = len(self.prompt_suffix.split())
        self.max_len_input_text = int(
            (self.max_len_model - len_instruction*self.avg_tokens_per_en_word - len_output*self.avg_tokens_per_en_word) / self.avg_tokens_per_word_avg
            )

        # define the role of the system in the conversation
        self.system_role = gpt_system_role
        # load environment variables
        load_dotenv('.env')
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError), max_tries=5)
    def completions_with_backoff(**kwargs):
        return openai.Completion.create(**kwargs) 

    @staticmethod
    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError), max_tries=5)
    def chat_completions_with_backoff(**kwargs):
        return openai.ChatCompletion.create(**kwargs) 

    def generate_predictions(
            self,
            input_texts,
            sleep_after_step=0,
            ):
        """
        Generate predictions for the input texts using an OpenAI language model.
        """

        prompts = []
        predictions = []

        # Generate a prompt and a prediction for each input text
        for i, input_text in enumerate(input_texts):
            # Create the prompt
            prompt = f'{self.instruction} {input_text} {self.prompt_suffix}'

            # if prompt is longer then max_len_model, remove words from the imput text
            len_prompt = int(len(prompt.split())*self.avg_tokens_per_word_avg)
            if len_prompt > self.max_len_model:
                # remove words from the input text
                input_text = input_text.split()
                input_text = input_text[:self.max_len_input_text]
                input_text = ' '.join(input_text)
                prompt = f'{self.instruction} {input_text} {self.prompt_suffix}'

                # print detailed info about the above operation
                logger.info(
                    f'Prompt n.{i} was too long, so we removed words from it. '
                    f'Approx original length: {len_prompt}; '
                    f'Approx new length: {int(len(prompt.split())*self.avg_tokens_per_word_avg)}'
                    )

            # log first prompt
            logger.info(prompt) if i == 0 else None

            # Print progress every 100 sentences
            if (i+1) % 20 == 0:
                logger.info(f"Processed {i+1} sentences")

            # Add the prompt to the list of prompts
            prompts.append(prompt)

            # call OpenAI's API to generate predictions
            try:
                # use chat completion for GPT3.5/4 models
                if self.model_name.startswith('gpt'):
                    gpt_out = self.chat_completions_with_backoff(
                        model=self.model_name,
                        messages=[
                            {"role": "system","content": self.system_role},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=15,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    # Extract the predicted label from the output
                    predicted_label = gpt_out['choices'][0]['message']['content'].strip()

                    # Save predicted label to file, together with the index of the prompt
                    with open('raw_predictions_cache.txt', 'a') as f:
                        f.write(f'{i}\t{predicted_label}\n')

                    # Sleep in order to respect OpenAPI's rate limit
                    time.sleep(sleep_after_step)

                # use simple completion for GPT3 models (text-davinci, etc.)
                else:
                    gpt_out = self.completions_with_backoff(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=0,
                        max_tokens=15,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    # Extract the predicted label from the output
                    predicted_label = gpt_out['choices'][0]['text'].strip()

            # manage API errors
            except Exception as e:
                logger.error(f'Error in generating prediction for prompt n.{i}: {e}')
                # since the prediction was not generated, use the default label
                predicted_label = self.default_label
                logger.warning(f'Selected default label "{predicted_label}" for prompt n.{i}.')

            # Add the predicted label to the list of predictionss
            predictions.append(predicted_label)

        return prompts, predictions

class HFClassifier(LMClassifier):
    def __init__(
            self,
            labels_dict,
            label_dims,
            default_label,
            instruction,
            prompt_suffix,
            model_name,
            max_len_model,
            output_dir=None,
            cache_dir=None,
            **kwargs,
            ):
                
        super().__init__(labels_dict, label_dims, default_label, instruction, prompt_suffix, model_name, max_len_model, output_dir, **kwargs)

        # Set device
        self.device = 'GPU' if torch.cuda.is_available() else 'CPU'
        logger.info(f'Running on {self.device} device...')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", cache_dir=cache_dir)

    def generate_predictions(self, input_texts):
        """
        Generate predictions for the input texts using an HuggingFace language model.
        """

        # Encode the labels
        encoded_labels = self.tokenizer(list(self.labels_dict.keys()), padding=True, truncation=True, return_tensors="pt")['input_ids']
        logger.info(f'Encoded labels: \n{encoded_labels}')

        # Retrieve the tokens associated to encoded labels and print them
        # decoded_labels = tokenizer.batch_decode(encoded_labels)
        # print(f'Decoded labels: \n{decoded_labels}')
        max_len = max(encoded_labels.shape[1:])
        logger.info(f'Maximum length of the encoded labels: {max_len}')

        predictions = []
        prompts = []

        # Generate a prompt and a prediction for each input text
        for i, input_text in enumerate(input_texts):
            # Create the prompt
            prompt = f'{self.instruction} {input_text} {self.prompt_suffix}'

            # log first prompt
            logger.info(prompt) if i == 0 else None

            # Print progress every 100 sentences
            if (i+1) % 100 == 0:
                logger.info(f"Processed {i+1} sentences")

            # Add the prompt to the list of prompts
            prompts.append(prompt)

            # Activate inference mode
            torch.inference_mode(True)
            
            # Encode the prompt using the tokenizer and generate a prediction using the model
            with torch.no_grad():

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                # If inputs is longer then max_len_model, remove tokens from the encoded instruction
                len_inputs = inputs['input_ids'].shape[1]
                if len_inputs > self.max_len_model:
                    # get the number of tokens to remove from the encoded instruction
                    len_remove = len_inputs - self.max_len_model

                    # get the length of the output
                    len_output = self.tokenizer(self.prompt_suffix, return_tensors="pt")['input_ids'].shape[1] + 1 # +1 for the full stop token

                    # remove inputs tokens that come before the output in the encoded prompt
                    inputs['input_ids'] = torch.cat((inputs['input_ids'][:,:-len_remove-len_output], inputs['input_ids'][:,-len_output:]),dim=1)
                    inputs['attention_mask'] = torch.cat((inputs['attention_mask'][:,:-len_remove-len_output], inputs['attention_mask'][:,-len_output:]),dim=1)
                    
                    # print info about the truncation
                    logger.info(f'Original input text length: {len_inputs}. Input has been truncated to {self.max_len_model} tokens.')
                
                # Generate a prediction
                outputs = self.model.generate(**inputs, max_new_tokens=max_len) # or max_length=inputs['input_ids'].shape[1]+max_len
                predicted_label = self.tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
                predictions.append(predicted_label)

            # Clear the cache after each iteration
            torch.cuda.empty_cache()

        """
        # Lowercase the predictions
        predictions =  list(map(str.lower,predictions))

        # TODO: Map the predictions to the labels (or empty string if label not found)
        # for now, just use the predictions as is. If only a substring equals a label, it does not get mapped to the label.
        predictions = [self.labels_dict.get(word) for word in predictions]


        # Add the data to the DataFrame
        df = pd.DataFrame({'time': time, 'prompt': prompts, 'prediction': predictions})
        """

        return prompts, predictions