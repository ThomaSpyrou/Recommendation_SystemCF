import pandas as pd
import numpy as np
import random


def read_merged_csv_file():
    """
    Read the new csv file.
    :return: dataframe all data from CSV
    """
    try:
        all_data = pd.read_csv('all_ratings_merged.csv', delimiter='\t', encoding='utf-8')
        books = pd.read_csv('books.csv', delimiter='\t', encoding='utf-8')

        return all_data, books

    except AttributeError:
        print("CSV files are not in proper format.")


def recommend(merged_data, books):
    """
    :param merged_data: dataframe all the data merged
    :return:
    """
    random_indexes = random.sample(range(0, len(merged_data)), 5)
    users_id = []
    for item in random_indexes:
        users_id.append(merged_data.iloc[item].userId)

    users = (merged_data.loc[merged_data['userId'].isin(users_id)])
    users = users.sort_values(by=['userId', 'bookRating'], ascending=[True, False])  # sort them in order to the top exist the biggest rate
    users_after_reduction = users.groupby('userId', group_keys=False).apply(lambda c: c.nlargest(3, 'bookRating')) # choose the first 3 book ratings for each user

    recommend_list = []
    # [ userId : { keywords: values, authors: values, year: values}]


if __name__ == '__main__':
    merged_data, books = read_merged_csv_file()
    recommend(merged_data, books)
