"""
python version: 3.6.9
pip version: 19.0.3
To run you have install needed modules
    1)pip3 install pandas
    2)pip3 install nltk
"""

import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize


def read_data():
    """
    Read the data from CSV file. The type of the data is Dataframe pandas. Set used columns.
    :return: The data from the 3 CSV in 3 different Dataframe variables.
    """
    try:

        book_ratings = pd.read_csv('BX-CSV-Dump/BX-Book-Ratings.csv', delimiter=';', encoding='latin1')
        book_ratings.columns = ['userId', 'ISBN', 'bookRating']

        users = pd.read_csv('BX-CSV-Dump/BX-Users.csv', delimiter=';', encoding='latin1')
        users.columns = ['userId', 'location', 'age']

        books = pd.read_csv('BX-CSV-Dump/BX-Books.csv', delimiter=";", error_bad_lines=False, encoding='latin1')
        books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageURLS', 'imageURLM',
                         'imageURLL']
        books.drop(['imageURLS', 'imageURLM', 'imageURLL'], axis=1, inplace=True)

        return book_ratings, users, books

    except AttributeError:
        print("CSV files are not in proper format.")


def pre_processing(book_ratings, users, books):
    """
    Delete books with less than 10 ratings and  users that have rate less than 5 books.
    :param book_ratings: dataframe (pd)
    :param users: dataframe (pd)
    :param books: dataframe (pd)
    :return: processed data
    """
    counter_of_users = book_ratings['userId'].value_counts()
    counter_of_ratings = book_ratings['ISBN'].value_counts()

    # find books which have less than 10 ratings
    books_to_delete = []
    for index, value in counter_of_ratings.items():
        if int(value) < 10:
            books_to_delete.append(index)

    # find user who have rate less than 5 books
    users_to_delete = []
    for index, value in counter_of_users.items():
        if int(value) < 5:
            users_to_delete.append(index)


if __name__ == '__main__':
    book_ratings, users, books = read_data()
    pre_processing(book_ratings, books, users)
