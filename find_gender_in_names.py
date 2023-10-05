"""
Example usage:

python find_gender_in_names.py \
    --name_gender_table data/gender_classification/gender_firstnames_ITA_processed.csv \
    --db_file /scratch/mentalism/data/database/MENTALISM_update.db \
    --table_name user_geocoded \
    --new_table_name user_geocoded \
    --chunk_size 50000 \
    --remove_columns [male_name,female_name] \
    --json_out data/gender_classification/user_gender.csv
"""

import sqlite3
import pandas as pd
from tqdm import tqdm
import fire

USER_GEOCODED_COLUMN_NAMES = [
    "user_id",
    "username",
    "full_name",
    "location",
    "join_year",
    "join_month",
    "join_day",
    "bio",
    "tweets",
    "following",
    "followers",
    "likes",
    "male_name",
    "female_name",
    "loc_count",
    "location_clean",
    "foreign_country",
    "all_regions",
    "region_pos",
    "region",
    "term_for_italy",
    "name_city_engl",
    "condition",
    "city_id",
    "all_cities",
    "city_pos",
    "region_code",
]


USER_COLUMN_NAMES = [
    "user_id",
    "username",
    "full_name",
    "location",
    "join_year",
    "join_month",
    "join_day",
    "bio",
    "tweets",
    "following",
    "followers",
    "likes",
    "male_name",
    "female_name",
]

class NameToGender():
    def __init__(self, name_gender_table:str) -> None:
        # read table of names and their associated 
        self.df_name_gender = pd.read_csv(name_gender_table, sep=',')

    def is_name_male(self, row):
        name = row['full_name'].split(' ')[0].lower()
        is_male = self.df_name_gender.query(f"name=='{name}'")["is_male"]
        return is_male.values[0] if len(is_male) > 0 else None

def main(
        db_file: str,
        table_name: str,
        new_table_name: str,
        chunk_size: int,
        name_gender_table:str,
        remove_columns:list[str]=['male_name', 'female_name'],
        json_out:str=None
    ):

    # Instantiate name to gender function
    class_to_apply = NameToGender(name_gender_table)

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

    column_names = USER_COLUMN_NAMES if table_name == 'user' else USER_GEOCODED_COLUMN_NAMES

    # Loop through the data in chunks
    for offset in range(0, total_rows, chunk_size):
        # Query the database for a chunk of rows
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset}")
        rows = cursor.fetchall()

        # Create a DataFrame from the fetched rows
        chunk_df = pd.DataFrame(rows, columns=column_names)
        
        # Remove the 'male_name' and 'female_name' columns
        chunk_df = chunk_df.drop(columns=remove_columns)

        # Apply the label function and update the "is_male" column
        chunk_df["is_male"] = chunk_df.apply(class_to_apply.is_name_male, axis=1)

        # Append to the result DataFrame
        result_df = pd.concat([result_df, chunk_df], ignore_index=True)
        
        # Update the progress bar
        progress_bar.update(len(rows))

    # Close the tqdm progress bar
    progress_bar.close()

    # Rename old table if needed
    if new_table_name == table_name:
        cursor.execute(f"ALTER TABLE {table_name} RENAME TO {table_name+'_old'}")

    # Create a new table including this script's results
    result_df.to_sql(new_table_name, conn, index=False, if_exists='fail')

    # Close the database connection
    conn.close()

    # Filter out rows with None gender labels
    result_df = result_df.dropna(subset=["is_male"])

    # Save results to json
    result_df.to_json(json_out, orient='records')

if __name__ == '__main__':
    fire.Fire(main)