"""
Example usage:

python enrich_user_id.py \
    --user_ids_file data/user_classification/user_age_gender_location.pkl \
    --db_file ../mentalism/sentemb/mydata/database/myMENTALISM.db \
    --table_name sample_tweets \
    --data_chunk_size 10000000 \
    --user_ids_chunk_size 11000 \
    --max_rows_per_user 100 \
    --fout .

python enrich_user_id.py \
--user_ids_file ../mentalism/sentemb/mydata/database/myMENTALISM.db \
--user_ids_table sample_tweets \
--db_file ../mentalism/sentemb/mydata/database/myMENTALISM.db \
--table_name sample_tweets \
--data_chunk_size 10000000 \
--user_ids_chunk_size 11000 \
--max_rows_per_user 100 \
--fout .

python enrich_user_id.py \
    --user_ids_file /scratch/mentalism/data/user_classification/user_age_gender_location.pkl \
    --db_file /scratch/mentalism/data/database/MENTALISM.db \
    --table_name tweets \
    --data_chunk_size 10000000 \
    --user_ids_chunk_size 11000 \
    --max_rows_per_user 100 \
    --remove_columns None \
    --fout /scratch/mentalism/data/user_classification/tweets_by_user_id_v2

db=/g100_work/IscrC_mental/data/database/MENTALISM.db
python enrich_user_id.py \
    --user_ids_file $db \
    --db_file  $db \
    --user_ids_table user_regioncoded \
    --table_name tweets \
    --data_chunk_size 10000000 \
    --user_ids_chunk_size 100000 \
    --max_rows_per_user 100 \
    --remove_columns None \
    --fout /scratch/mentalism/data/user_classification/tweets_regioncoded_users
"""
import fire
import os
import sqlite3
import pandas as pd
from tqdm import tqdm


TABLE_COLUMN_NAMES = [
    "tweet_id",
    "user_id",
    "created_at",
    "text",
    "retweet_text",
]

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

def get_user_ids(db_file, table_name):
    # Create a database connection
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    try:
        # Query unique user IDs from the specified table
        cursor.execute(f"SELECT DISTINCT user_id FROM {table_name}")
        user_ids = [row[0] for row in cursor.fetchall()]
        return user_ids

    except sqlite3.Error as e:
        print(f"Error reading data from the database: {e}")

    finally:
        # Close the database connection
        conn.close()

def get_info_by_user_ids(
        user_ids_file,
        db_file,
        table_name,
        data_chunk_size,
        user_ids_chunk_size,
        max_rows_per_user,
        column_names=TABLE_COLUMN_NAMES,
        remove_columns=None,
        user_ids_table='user_geocoded',
        fout='.',
        ):
    
    # Read the user ids to retrieve
    if user_ids_file.endswith('.pkl'):
        user_ids_to_retrieve = pd.read_pickle(user_ids_file).user_id.astype(int).tolist()
    elif user_ids_file.endswith('.csv'):
        user_ids_to_retrieve = pd.read_csv(user_ids_file).user_id.astype(int).tolist()
    elif user_ids_file.endswith('.db') and user_ids_table:
        user_ids_to_retrieve = get_user_ids(user_ids_file, user_ids_table)
    else:
        raise NotImplementedError
    
    print(f'Found {len(user_ids_to_retrieve)} user ids to retrieve')

    # Split user_ids_to_retrieve in chunks of max_user_ids_per_file
    user_ids_to_retrieve_chunks = [
        user_ids_to_retrieve[i:i + user_ids_chunk_size] for i in range(0, len(user_ids_to_retrieve), user_ids_chunk_size)
        ]
    
    # Create a database connection
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Get the total number of rows
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = cursor.fetchone()[0]

    # Initialize an empty DataFrame
    result_df = pd.DataFrame()

    n_saved_files=0
    # Loop through user_ids_to_retrieve in chunks
    for n_chunk, user_ids_to_retrieve in enumerate(user_ids_to_retrieve_chunks):
        # Loop through the data in chunks
        # Initialize a tqdm progress bar
        progress_bar = tqdm(total=total_rows, unit="row", desc="Processing")
        for offset in range(0, total_rows, data_chunk_size):
            # Query the database for a chunk of rows
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {data_chunk_size} OFFSET {offset}")
            rows = cursor.fetchall()

            # Create a DataFrame from the fetched rows
            chunk_df = pd.DataFrame(rows, columns=column_names)
            
            # Remove the unwanted columns
            if remove_columns is not None:
                chunk_df = chunk_df.drop(columns=remove_columns)

            # Append to the result DataFrame
            chunk_df = chunk_df[chunk_df['user_id'].isin(user_ids_to_retrieve)]
            result_df = pd.concat([result_df, chunk_df[chunk_df['user_id'].isin(user_ids_to_retrieve)]], ignore_index=True)

            # Sort the result DataFrame by user_id and created_at
            result_df = result_df.sort_values(by=['user_id','created_at'], ascending=False)
            # Keep only the first max_tweets_per_user tweets for each user
            result_df = result_df.groupby('user_id').head(max_rows_per_user)
            
            # Update the progress bar
            progress_bar.update(len(rows))

        # Save results to pickle file
        print(f'Saving {len(result_df)} tweets to user_tweets_chunk{n_chunk}.pkl')
        result_df.to_pickle(os.path.join(fout, f'user_tweets_chunk{n_chunk}.pkl'))
        # Reset the result DataFrame
        result_df = pd.DataFrame()

    # Close the tqdm progress bar
    progress_bar.close()

    # Close the database connection
    conn.close()

if __name__ == '__main__':
    fire.Fire(get_info_by_user_ids)
