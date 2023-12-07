import os
import pandas as pd
from fire import Fire
from task_manager import TaskManager
from classifiers import HFClassifier
from evaluate import evaluate_predictions
from utils import incremental_path, setup_logging
from logging import getLogger
logger = getLogger(__name__)

def classify_and_evaluate(
        data_file,
        task_file,
        model_name,
        max_len_model,
        output_dir,
        batch_size=32,
        cache_dir=None,
        evaluation_only=False,
        log_to_file=True,
        ):
    """
    Params:
        data_file: path to the data file
        task_file: path to the task file
        model_name: name of the model to use (for HuggingFace models, use the full name, e.g. "username/model_name")
        max_len_model: maximum input length of the model
        output_dir: path to the output directory
        batch_size: batch size for the model
        cache_dir: path to the cache directory, where to store/load the HF model
        evaluation_only: if True, only evaluate the predictions that are already present in the output_dir
    Output:
        predictions.csv: csv file with the predictions and the probabilities for each class
        *.log: log files with the logs from the predictions process and the evaluation of the predictions
    """

    # Duplicate the output to stdout and a log file
    # strip points and slashes from the model name
    model_name_short = model_name.split("/")[-1].replace(".", "") # remove "username/" in case of HF models

    output_base_dir = f"{output_dir}/{model_name_short}"
    output_dir = incremental_path(output_base_dir, select_last=evaluation_only)

    setup_logging(os.path.basename(__file__).split('.')[0], logger, output_dir if log_to_file else None)

    logger.info(f'Working on {output_dir}')

    # Define task and load data
    tm = TaskManager(task_file)
    logger.info(f'Loading data...')
    input_texts, gold_labels = tm.read_data(data_file)

    if not evaluation_only:
        # tokenize features
        classifier = HFClassifier(
            model_name=model_name,
            max_len_model=max_len_model,
            batch_size=batch_size,
            output_dir=output_dir,
            cache_dir=cache_dir,
            log_to_file=log_to_file
        )

        # classifier.initialize_dataloader(input_texts)

        logger.info(f'Generating predictions...')
        predictions, class_probs = classifier.generate_predictions(input_texts)

        # Save predictions
        probs_columns_names = [f'prob_class{i}' for i in range(class_probs.shape[-1])]
        df_preds = pd.DataFrame(predictions, columns=['prediction'])
        df_probs = pd.DataFrame(class_probs, columns=probs_columns_names)
        df = pd.concat([df_preds, df_probs], axis=1).set_index(input_texts.index)
        df.to_csv(os.path.join(output_dir, f'predictions.csv'))
    else:
        # load predictions
        df_preds = pd.read_csv(os.path.join(output_dir, f'predictions.csv'))[['prediction']]
    
    # Evaluate
    logger.info(f'Evaluating...')
    df_preds['prediction'] = df_preds['prediction'].astype(str).apply(lambda x: tm.labels[x])
    if gold_labels is not None:
        evaluate_predictions(df_preds, gold_labels, logdir=output_dir)

if __name__ == "__main__":
    Fire(classify_and_evaluate)




