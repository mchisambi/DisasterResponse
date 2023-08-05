import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function that combines two separate datasets for messages and categories and returns a merged dataframe.
    
    Inputs:
        messages_filepath: Filepath of the messages dataset.
        categories_filepath: Filepath of the categories dataset.
    
    Returns:
        df: Merged dataframe containing both datasets.
    '''
    # Read categories and messages datasets
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    
    # Merge datasets on 'id'
    df = pd.merge(categories, messages, how='inner', on=['id'], left_index=True)
    
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
   
    # Select the first row of the categories dataframe
    row = categories.loc[0]
    
    # Extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    # Adapt values to be binary
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Drop the original 'categories' column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    combination = [df, categories]
    df = pd.concat(combination, axis=1)
    
    return df


def clean_data(df):
    '''
    Takes a dataframe and drops duplicate rows.
    
    Input: 
        df: Dataframe containing the merged content of messages and categories datasets.
    
    Returns: 
        df: Dataframe without duplicate rows.
    '''
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Saves the dataframe to an SQLite database.
    
    Inputs:
        df: Dataframe containing the cleaned version of the merged message and categories data.
        database_filename: Filename for the output database.
    
    Returns:
        None
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to the database!')
    
    else:
        print('Please provide the filepaths of the messages and categories datasets as the first and second argument respectively, as well as the filepath of the database to save the cleaned data to as the third argument. \n\nExample: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')


if __name__ == '__main__':
    main()
