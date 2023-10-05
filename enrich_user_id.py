"""
Example usage:

python enrich_user_id.py \
    --user_ids_table ../../../pappa/data/user_classification/user_age_gender_location.pkl \
    --db_file ../mydata/database/myMENTALISM.db \
    --table_name sample_tweets \
    --chunk_size 10000000 \
    --max_tweets_in_file 1000000 \
    --remove_columns None

python enrich_user_id.py \
    --user_ids_table /scratch/mentalism/data/user_classification/user_age_gender_location.pkl \
    --db_file /scratch/mentalism/data/database/MENTALISM_update.db \
    --table_name tweets \
    --chunk_size 10000000 \
    --max_tweets_in_file 1000000 \
    --remove_columns None \
    --fout /scratch/mentalism/data/user_classification/tweets_by_user_id_v2
    
"""
import fire
import sqlite3
import pandas as pd
from tqdm import tqdm


TABLE_COLUMN_NAMES =[
    'tweet_id',
    'user_id',
    'created_at',
    'text',
    'retweet_text',
    'language',
    'likes',
    'retweets'
    ]

# define a main function that accomplishs the above
def get_info_by_user_ids(
        user_ids_table,
        db_file,
        table_name,
        chunk_size,
        column_names=TABLE_COLUMN_NAMES,
        remove_columns=None,
        max_tweets_in_file=1000000,
        fout=None,
        ):
    
    # Read the user ids to retrieve
    user_ids_to_retrieve = pd.read_pickle(user_ids_table).user_id.astype(int).tolist()
    
    # Create a database connection
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Get the total number of rows
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = cursor.fetchone()[0]

    # Initialize an empty DataFrame
    result_df = pd.DataFrame()

    # Initialize a tqdm progress bar
    progress_bar = tqdm(total=total_rows, unit="row", desc="Processing")

    n_saved_files=0
    # Loop through the data in chunks
    for offset in range(0, total_rows, chunk_size):
        # Query the database for a chunk of rows
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset}")
        rows = cursor.fetchall()

        # Create a DataFrame from the fetched rows
        chunk_df = pd.DataFrame(rows, columns=column_names)
        
        # Remove the unwanted columns
        if remove_columns is not None:
            chunk_df = chunk_df.drop(columns=remove_columns)

        # Append to the result DataFrame
        # chunk_df['user_id'] = chunk_df['user_id'].astype(str)
        result_df = pd.concat([result_df, chunk_df[chunk_df['user_id'].isin(user_ids_to_retrieve)]], ignore_index=True)

        if len(result_df) >= max_tweets_in_file:
            # Save results to pickle
            print(f'Saving {len(result_df)} tweets to file {n_saved_files}')
            result_df.to_pickle(f'{fout}_{n_saved_files}.pkl')
            # Reset the result DataFrame
            result_df = pd.DataFrame()
            n_saved_files += 1
        
        # Update the progress bar
        progress_bar.update(len(rows))

    # Close the tqdm progress bar
    progress_bar.close()

    # Close the database connection
    conn.close()

if __name__ == '__main__':
    fire.Fire(get_info_by_user_ids)
