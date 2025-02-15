# basic libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """joins training messages and classifications according to their id into a dataframe"""
    messages =pd.read_csv(messages_filepath,encoding='utf-8')
    categories = pd.read_csv(categories_filepath,encoding='utf-8')

    return pd.merge(messages,categories,"inner",on="id")

def clean_data(df):
    """produces clean data out of the joined dataframe (of messages and categories) """
    categories=pd.Series(df.categories).str.split(";",expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0:1,:]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [row.iloc[0:1,i].values[0][0:-2] for i in range(0,row.shape[1])]
    # rename the columns of `categories`
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = list(map(lambda x:x[-1:],pd.Series(categories[column].astype(str)).values) )

        # convert column from string to numeric
        categories[column] = pd.Series(categories[column].astype(int)).values

        # allow only 0 and 1; anything that is not 1 is assumed as 0.
        categories[column] = list(map(lambda x:(1 if x==1 else 0),pd.Series(categories[column]).values) )


    # drop the original categories column from `df`
    df.drop(columns=['categories'],index=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df=pd.concat([df,categories],axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    """saves dataframe "df" as a database table "mytable" in the file "database_filename" """
    engine = create_engine("sqlite:///"+database_filename)
    df.to_sql('mytable', engine, index=False, if_exists='replace')


def main():
    """procedure covering all steps of the ETL pipeline"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
